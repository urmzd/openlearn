-- Mart (gold): the business-facing table a BI tool queries. Materialized as a
-- real table for fast dashboards. One grain: one row per (order_date, currency).
with enriched as (

    select * from {{ ref('int_orders_enriched') }}

)

select
    order_date,
    currency,
    count(distinct order_id)        as orders,
    count(distinct customer_id)     as customers,
    sum(recognized_revenue)         as revenue

from enriched
group by order_date, currency
