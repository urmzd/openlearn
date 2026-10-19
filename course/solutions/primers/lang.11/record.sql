-- lang.11 reference: record one request and its hourly rollup atomically and
-- idempotently. Parameters are the usage columns (:request_id, :ts_ms, ...).
BEGIN IMMEDIATE;
INSERT INTO usage (request_id, ts_ms, tenant, key_id, model, served_model, route, api_version,
                   status, error_code, stream, prompt_tokens, completion_tokens, cached_tokens,
                   ttft_ms, e2e_ms, cache_hit, worker_id, trace_id)
VALUES (:request_id, :ts_ms, :tenant, :key_id, :model, :served_model, :route, :api_version,
        :status, :error_code, :stream, :prompt_tokens, :completion_tokens, :cached_tokens,
        :ttft_ms, :e2e_ms, :cache_hit, :worker_id, :trace_id)
ON CONFLICT (request_id) DO NOTHING;
-- changes() is the number of rows the previous statement inserted: 0 when
-- the request id was already recorded, so a replay adds nothing.
INSERT INTO usage_hourly (tenant, hour_ms, requests, prompt_tokens, completion_tokens)
SELECT :tenant, :ts_ms - :ts_ms % 3600000, 1, :prompt_tokens, :completion_tokens
WHERE changes() = 1
ON CONFLICT (tenant, hour_ms) DO UPDATE SET
    requests          = requests + 1,
    prompt_tokens     = prompt_tokens + excluded.prompt_tokens,
    completion_tokens = completion_tokens + excluded.completion_tokens;
COMMIT;
