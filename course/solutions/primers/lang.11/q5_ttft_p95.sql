-- lang.11 reference: nearest-rank p95 of TTFT per model in [:since, :until):
-- the value at rank ceil(0.95 n) in ascending order, NULL TTFTs left out.
WITH ranked AS (
    SELECT model, ttft_ms,
           ROW_NUMBER() OVER (PARTITION BY model ORDER BY ttft_ms) AS rn,
           COUNT(*)     OVER (PARTITION BY model)                  AS n
    FROM usage
    WHERE ttft_ms IS NOT NULL AND ts_ms >= :since AND ts_ms < :until
)
SELECT model, n, ttft_ms AS p95_ttft_ms
FROM ranked
WHERE rn = (95 * n + 99) / 100
ORDER BY model;
