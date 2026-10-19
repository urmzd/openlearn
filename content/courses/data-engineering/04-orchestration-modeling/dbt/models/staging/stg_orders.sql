-- Staging: the contract boundary with raw data. One model per source, doing
-- only light rename/recast -- no business logic. Materialized as a view (cheap,
-- always fresh). Everything downstream references THIS, never the source.
with source as (

    select * from {{ source('silver', 'orders') }}

),

renamed as (

    select
        order_id,
        customer_id,
        status      as order_status,
        amount      as order_amount,
        currency,
        event_ts    as ordered_at,
        order_date

    from source

)

select * from renamed
