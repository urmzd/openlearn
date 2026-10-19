-- lang.11 reference: one tenant's usage per model in [:since, :until).
SELECT model,
       COUNT(*)                                        AS requests,
       SUM(status >= 400 OR error_code IS NOT NULL)    AS errors,
       SUM(prompt_tokens)                              AS prompt_tokens,
       SUM(completion_tokens)                          AS completion_tokens
FROM usage
WHERE tenant = :tenant AND ts_ms >= :since AND ts_ms < :until
GROUP BY model
ORDER BY model;
