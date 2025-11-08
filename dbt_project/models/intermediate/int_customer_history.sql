{{
  config(
    materialized='view'
  )
}}

with loans as (
    select * from {{ ref('stg_loans') }}
),

customer_aggregates as (
    select
        member_id,

        -- Loan counts
        count(*) as total_loans,
        count(case when is_default = 1 then 1 end) as total_defaults,
        count(case when is_default = 0 then 1 end) as total_paid,

        -- Default rate
        case
            when count(*) > 0
            then cast(count(case when is_default = 1 then 1 end) as float) / count(*)
            else 0
        end as historical_default_rate,

        -- Total amounts
        sum(loan_amount) as total_borrowed,
        avg(loan_amount) as avg_loan_amount,
        max(loan_amount) as max_loan_amount,
        min(loan_amount) as min_loan_amount,

        -- Interest rates
        avg(interest_rate) as avg_interest_rate,
        max(interest_rate) as max_interest_rate,

        -- Credit score trend
        avg(fico_score_avg) as avg_fico_score,
        min(fico_score_avg) as min_fico_score,
        max(fico_score_avg) as max_fico_score,

        -- DTI
        avg(debt_to_income) as avg_dti,
        max(debt_to_income) as max_dti,

        -- First and last loan dates
        min(issue_date) as first_loan_date,
        max(issue_date) as most_recent_loan_date,

        -- Tenure calculation
        case
            when min(issue_date) is not null and max(issue_date) is not null
            then (julianday(max(issue_date)) - julianday(min(issue_date))) / 30.0
            else 0
        end as customer_tenure_months

    from loans
    where member_id is not null
    group by member_id
)

select * from customer_aggregates
