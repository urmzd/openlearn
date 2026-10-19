// Package ledger is the gateway's usage ledger (gw.07): one SQLite row per
// metered request (formats/usage.v1.sql), the meter stage of the middleware
// chain that writes those rows (meter.go), and the admin API under
// /admin/v1 whose server side this module owns (admin.go).
//
// The store is SQLite in WAL mode through the pure-Go driver
// modernc.org/sqlite (contracts/allowed-deps.toml), so the gateway binary
// needs no C toolchain and a SIGKILL leaves every row complete or absent.
//
// Contract: formats/usage.v1.sql (the schema, applied verbatim),
// openapi/admin.v1.yaml (GET /admin/v1/usage, UsageRow). Chapter:
// content/tracks/ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md.
package ledger

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver
)

// Schema is contracts/formats/usage.v1.sql, byte for byte. Open applies it;
// applying it twice is a no-op (every statement is IF NOT EXISTS or OR IGNORE).
const Schema = `-- contracts/formats/usage.v1.sql: the gateway's usage ledger.
-- SQLite, at [gateway].usage_db (/artifacts/gateway/usage.db), WAL mode, so
-- a SIGKILL leaves every row complete or absent (gw.07).
--
-- modules: gw.07 (owns it: writes rows, serves GET /admin/v1/usage),
-- ag.04 (the query_usage tool reads it through the SQL safety gate),
-- craft.14 (api_version), ops.09 (per-tenant usage)
--
-- One row per request that reached the handler chain past authentication,
-- including rejected ones (status >= 400), written when the response ends.
-- Token counts come from the engine's usage (the final SSE usage chunk or
-- the response body); cached_tokens from usage.prompt_tokens_details in v2,
-- else 0. Times are Unix milliseconds. Applying this file twice is a no-op.

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL PRIMARY KEY
);
INSERT OR IGNORE INTO schema_version (version) VALUES (1);

CREATE TABLE IF NOT EXISTS usage (
    request_id        TEXT    NOT NULL PRIMARY KEY,  -- X-Request-Id
    ts_ms             INTEGER NOT NULL,              -- request start
    tenant            TEXT    NOT NULL,
    key_id            TEXT    NOT NULL,              -- the <id> of tl_<id>_<secret>
    model             TEXT    NOT NULL,              -- the public model id the client sent
    served_model      TEXT    NOT NULL DEFAULT '',   -- the backend that served it ('' when rejected)
    route             TEXT    NOT NULL,              -- /v1/chat/completions, /v1/completions, /v1/embeddings
    api_version       TEXT    NOT NULL DEFAULT '1',  -- '1' or '2' (openai-subset.v2.yaml)
    status            INTEGER NOT NULL,              -- HTTP status sent to the client
    error_code        TEXT,                          -- the error's code, NULL on success
    stream            INTEGER NOT NULL CHECK (stream IN (0, 1)),
    prompt_tokens     INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
    cached_tokens     INTEGER NOT NULL DEFAULT 0 CHECK (cached_tokens >= 0 AND cached_tokens <= prompt_tokens),
    ttft_ms           REAL,                          -- NULL when no content byte was sent
    e2e_ms            REAL    NOT NULL,
    cache_hit         INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0, 1)),
    worker_id         TEXT    NOT NULL DEFAULT '',
    trace_id          TEXT    NOT NULL DEFAULT ''    -- 32 hex digits, '' when untraced
);

CREATE INDEX IF NOT EXISTS usage_tenant_ts ON usage (tenant, ts_ms);
CREATE INDEX IF NOT EXISTS usage_key_ts ON usage (key_id, ts_ms);
CREATE INDEX IF NOT EXISTS usage_model_ts ON usage (model, ts_ms);
`

// SchemaVersion is the version this code writes and reads.
const SchemaVersion = 1

// UsageRecord is one row of the usage table.
type UsageRecord struct {
	RequestID        string
	Start            time.Time // request start (ts_ms)
	Tenant           string
	KeyID            string
	Model            string // what the client asked for
	ServedModel      string // what the upstream answered with; "" when rejected
	Route            string
	APIVersion       string // "1" or "2"; "" means "1"
	Status           int
	ErrorCode        string // "" means NULL (success)
	Stream           bool
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
	TTFT             *time.Duration // nil when no content byte was sent
	E2E              time.Duration
	CacheHit         bool
	WorkerID         string
	TraceID          string
}

// UsageFilter selects rows: empty strings and zero times mean "any". Since
// is inclusive and Until exclusive (admin.v1.yaml).
type UsageFilter struct {
	Tenant  string
	KeyID   string
	Since   time.Time
	Until   time.Time
	GroupBy string // "", "none", "model", "tenant", "key_id", "api_version"
}

// UsageRow is admin.v1.yaml UsageRow: the group key (only the grouped field
// is set) and the sums of its rows. Errors counts rows with status >= 400 or
// an error code (a stream that failed after its first byte).
type UsageRow struct {
	Tenant           string `json:"tenant,omitempty"`
	KeyID            string `json:"key_id,omitempty"`
	Model            string `json:"model,omitempty"`
	APIVersion       string `json:"api_version,omitempty"`
	Requests         int64  `json:"requests"`
	Errors           int64  `json:"errors"`
	PromptTokens     int64  `json:"prompt_tokens"`
	CompletionTokens int64  `json:"completion_tokens"`
	CachedTokens     int64  `json:"cached_tokens"`
}

// Ledger is what the meter and the admin API need (gw.07).
type Ledger interface {
	Record(ctx context.Context, r UsageRecord) error
	Query(ctx context.Context, f UsageFilter) ([]UsageRow, error)
}

// Errors.
var (
	ErrInvalidRecord = errors.New("invalid usage record")
	ErrBadGroupBy    = errors.New("group_by must be one of none, model, tenant, key_id, api_version")
	ErrNewerSchema   = errors.New("the usage database was written by a newer schema version")
)

// groupColumns is the whitelist of GROUP BY columns: a column name cannot be
// a bound parameter, so it comes from this table and never from the caller.
var groupColumns = map[string]string{
	"model":       "model",
	"tenant":      "tenant",
	"key_id":      "key_id",
	"api_version": "api_version",
}

// DB is the SQLite ledger.
type DB struct {
	db *sql.DB
}

var _ Ledger = (*DB)(nil)

// DSN is the database/sql data source for path: every pooled connection gets
// busy_timeout (writers wait for the lock instead of failing with
// SQLITE_BUSY), WAL, and synchronous=NORMAL (in WAL mode a commit survives a
// process kill; only a power loss can take the last commits).
func DSN(path string) string {
	q := url.Values{}
	q.Add("_pragma", "busy_timeout(5000)")
	q.Add("_pragma", "journal_mode(WAL)")
	q.Add("_pragma", "synchronous(NORMAL)")
	return path + "?" + q.Encode()
}

// Open opens (creating if needed) the ledger at path and applies Schema. A
// database whose schema_version is newer than SchemaVersion is refused.
func Open(path string) (*DB, error) {
	db, err := sql.Open("sqlite", DSN(path))
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if _, err := db.ExecContext(ctx, Schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("apply usage.v1.sql to %s: %w", path, err)
	}
	var v int
	if err := db.QueryRowContext(ctx, "SELECT MAX(version) FROM schema_version").Scan(&v); err != nil {
		db.Close()
		return nil, err
	}
	if v > SchemaVersion {
		db.Close()
		return nil, fmt.Errorf("%w: %s has version %d, this gateway reads %d", ErrNewerSchema, path, v, SchemaVersion)
	}
	return &DB{db: db}, nil
}

// Close closes the database.
func (d *DB) Close() error {
	return d.db.Close()
}

func nullString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func ms(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

// Record writes one row in one statement, so it commits whole or not at all.
// It is idempotent by RequestID: a second record of the same request is
// ignored and the first one stands. Negative token counts are invalid;
// cached tokens are clamped to the prompt tokens.
func (d *DB) Record(ctx context.Context, r UsageRecord) error {
	if r.RequestID == "" || r.PromptTokens < 0 || r.CompletionTokens < 0 || r.CachedTokens < 0 {
		return fmt.Errorf("%w: request %q, tokens %d/%d/%d", ErrInvalidRecord, r.RequestID, r.PromptTokens, r.CompletionTokens, r.CachedTokens)
	}
	if r.CachedTokens > r.PromptTokens {
		r.CachedTokens = r.PromptTokens
	}
	if r.APIVersion == "" {
		r.APIVersion = "1"
	}
	var ttft any
	if r.TTFT != nil {
		ttft = ms(*r.TTFT)
	}
	b2i := func(b bool) int {
		if b {
			return 1
		}
		return 0
	}
	_, err := d.db.ExecContext(ctx, `INSERT INTO usage (
		request_id, ts_ms, tenant, key_id, model, served_model, route, api_version,
		status, error_code, stream, prompt_tokens, completion_tokens, cached_tokens,
		ttft_ms, e2e_ms, cache_hit, worker_id, trace_id)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT (request_id) DO NOTHING`,
		r.RequestID, r.Start.UnixMilli(), r.Tenant, r.KeyID, r.Model, r.ServedModel, r.Route, r.APIVersion,
		r.Status, nullString(r.ErrorCode), b2i(r.Stream), r.PromptTokens, r.CompletionTokens, r.CachedTokens,
		ttft, ms(r.E2E), b2i(r.CacheHit), r.WorkerID, r.TraceID)
	return err
}

// Query sums the matching rows, one UsageRow per group sorted by the group
// key; GroupBy "" or "none" gives exactly one row, zeros when nothing matches.
// Every value is a bound parameter.
func (d *DB) Query(ctx context.Context, f UsageFilter) ([]UsageRow, error) {
	col := ""
	if f.GroupBy != "" && f.GroupBy != "none" {
		c, ok := groupColumns[f.GroupBy]
		if !ok {
			return nil, ErrBadGroupBy
		}
		col = c
	}
	var where []string
	var args []any
	if f.Tenant != "" {
		where = append(where, "tenant = ?")
		args = append(args, f.Tenant)
	}
	if f.KeyID != "" {
		where = append(where, "key_id = ?")
		args = append(args, f.KeyID)
	}
	if !f.Since.IsZero() {
		where = append(where, "ts_ms >= ?")
		args = append(args, f.Since.UnixMilli())
	}
	if !f.Until.IsZero() {
		where = append(where, "ts_ms < ?")
		args = append(args, f.Until.UnixMilli())
	}
	sel := `COUNT(*),
		COALESCE(SUM(CASE WHEN status >= 400 OR error_code IS NOT NULL THEN 1 ELSE 0 END), 0),
		COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cached_tokens), 0)`
	q := "SELECT " + sel + " FROM usage"
	if col != "" {
		q = "SELECT " + col + ", " + sel + " FROM usage"
	}
	if len(where) > 0 {
		q += " WHERE " + strings.Join(where, " AND ")
	}
	if col != "" {
		q += " GROUP BY " + col + " ORDER BY " + col
	}
	rows, err := d.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []UsageRow{}
	for rows.Next() {
		var u UsageRow
		var key string
		dest := []any{&u.Requests, &u.Errors, &u.PromptTokens, &u.CompletionTokens, &u.CachedTokens}
		if col != "" {
			dest = append([]any{&key}, dest...)
		}
		if err := rows.Scan(dest...); err != nil {
			return nil, err
		}
		switch col {
		case "model":
			u.Model = key
		case "tenant":
			u.Tenant = key
		case "key_id":
			u.KeyID = key
		case "api_version":
			u.APIVersion = key
		}
		out = append(out, u)
	}
	return out, rows.Err()
}
