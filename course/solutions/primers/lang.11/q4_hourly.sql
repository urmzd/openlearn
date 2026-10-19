-- lang.11 reference: one tenant's requests and completion tokens per hour.
SELECT ts_ms - ts_ms % 3600000 AS hour_ms,
       COUNT(*)                AS requests,
       SUM(completion_tokens)  AS completion_tokens
FROM usage
WHERE tenant = :tenant AND ts_ms >= :since AND ts_ms < :until
GROUP BY hour_ms
ORDER BY hour_ms;
