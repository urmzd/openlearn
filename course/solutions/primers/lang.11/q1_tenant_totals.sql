-- lang.11 reference: one tenant's totals in [:since, :until).
SELECT COUNT(*)                                                             AS requests,
       COALESCE(SUM(status >= 400 OR error_code IS NOT NULL), 0)            AS errors,
       COALESCE(SUM(prompt_tokens), 0)                                      AS prompt_tokens,
       COALESCE(SUM(completion_tokens), 0)                                  AS completion_tokens
FROM usage
WHERE tenant = :tenant AND ts_ms >= :since AND ts_ms < :until;
