-- Intermediate: reusable business logic that more than one mart may want, kept
-- out of the marts so it is defined ONCE. `ref()` is the key idea -- dbt reads
-- this dependency on stg_orders, infers the DAG, and runs staging first.
with orders as (

    select * from {{ ref('stg_orders') }}

)

select
    order_id,
    customer_id,
    order_status,
    currency,
    ordered_at,
    order_date,
    order_amount,
    -- Revenue recognition rule, defined in exactly one place: only completed
    -- orders count. Every mart that needs "revenue" inherits this definition,
    -- which is how you kill metric drift across dashboards.
    case
        when order_status = 'completed' then order_amount
        else 0
    end as recognized_revenue

from orders
