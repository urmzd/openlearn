-- contracts/formats/usage.v1.sql: the gateway's usage ledger (DESIGN 2.9).
-- chapter: content/tracks/ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md
-- chapter: content/tracks/software-craftsmanship/12-language-and-tool-primers/11-sql-and-sqlite.md
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
