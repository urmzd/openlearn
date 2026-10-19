-- lang.11 reference: the :n keys that used the most tokens in [:since, :until).
SELECT key_id, tenant, SUM(prompt_tokens + completion_tokens) AS tokens
FROM usage
WHERE ts_ms >= :since AND ts_ms < :until
GROUP BY key_id, tenant
ORDER BY tokens DESC, key_id ASC
LIMIT :n;
