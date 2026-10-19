-- A SINGULAR test: any SELECT that should return zero rows. This is how you
-- encode a business invariant that the built-in tests can't express. It runs
-- with `dbt test` alongside the schema tests and fails the build on any row.
--
-- Invariants asserted here:
--   1. The grain holds: at most one row per (order_date, currency).
--   2. Revenue is never negative -- a negative day means a logic or data bug.

-- (1) grain violations: any key appearing more than once
select
    order_date,
    currency,
    count(*) as n_rows
from {{ ref('daily_revenue') }}
group by order_date, currency
having count(*) > 1

union all

-- (2) negative-revenue days
select
    order_date,
    currency,
    -1 as n_rows
from {{ ref('daily_revenue') }}
where revenue < 0
