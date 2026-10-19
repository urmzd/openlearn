package gate

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"

	"tinyllm/agent/tool"
)

// QueryUsageTool is the query_usage tool: one read-only SQL query over the
// gateway's usage ledger (formats/usage.v1.sql, table usage). The gate
// already classified the query; the tool classifies it again and runs it
// on a connection with PRAGMA query_only = ON, because the classifier is
// the weakest of the three layers (classifier, read-only connection, a
// SELECT-only role) and must never be the only one. db should be the
// tool's own handle on the ledger (opened read-only where possible).
//
// The result is JSON: {"columns": [...], "rows": [[...], ...]}, at most
// maxRows rows (DefaultMaxRows when 0).
func QueryUsageTool(db *sql.DB, maxRows int) tool.Tool {
	if maxRows <= 0 {
		maxRows = DefaultMaxRows
	}
	schema := `{"type":"object","properties":{"sql":{"type":"string","minLength":1,` +
		`"description":"One SQLite SELECT over table usage(request_id, ts_ms, tenant, key_id, model, served_model, route, api_version, status, error_code, stream, prompt_tokens, completion_tokens, cached_tokens, ttft_ms, e2e_ms, cache_hit, worker_id, trace_id). Times are Unix milliseconds."}},` +
		`"required":["sql"],"additionalProperties":false}`
	return tool.New("query_usage", "Read the gateway's usage ledger with one SQL SELECT.", schema,
		func(ctx context.Context, args json.RawMessage) (string, error) {
			var in struct{ SQL string }
			if err := json.Unmarshal(args, &in); err != nil {
				return "", err
			}
			r := ClassifySQL(in.SQL, maxRows)
			if !r.Allowed {
				return "", fmt.Errorf("refused: %s", r.Reason)
			}
			conn, err := db.Conn(ctx)
			if err != nil {
				return "", err
			}
			defer conn.Close()
			if _, err := conn.ExecContext(ctx, "PRAGMA query_only = ON"); err != nil {
				return "", err
			}
			rows, err := conn.QueryContext(ctx, r.SQL)
			if err != nil {
				return "", fmt.Errorf("query failed: %w", err)
			}
			defer rows.Close()
			cols, err := rows.Columns()
			if err != nil {
				return "", err
			}
			out := struct {
				Columns []string `json:"columns"`
				Rows    [][]any  `json:"rows"`
			}{Columns: cols, Rows: [][]any{}}
			for rows.Next() && len(out.Rows) < maxRows {
				vals := make([]any, len(cols))
				ptrs := make([]any, len(cols))
				for i := range vals {
					ptrs[i] = &vals[i]
				}
				if err := rows.Scan(ptrs...); err != nil {
					return "", err
				}
				for i, v := range vals {
					if b, ok := v.([]byte); ok {
						vals[i] = string(b)
					}
				}
				out.Rows = append(out.Rows, vals)
			}
			if err := rows.Err(); err != nil {
				return "", err
			}
			b, err := json.Marshal(out)
			return strings.TrimSpace(string(b)), err
		})
}
