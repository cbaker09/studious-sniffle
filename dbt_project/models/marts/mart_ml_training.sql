{{
  config(
    materialized='table'
  )
}}

with loan_features as (
    select * from {{ ref('int_loan_features') }}
),

customer_history as (
    select * from {{ ref('int_customer_history') }}
),

final_dataset as (
    select
        -- Primary key
        lf.loan_id,

        -- Target variable
        lf.is_default,

        -- Loan characteristics
        lf.loan_amount,
        lf.funded_amount,
        lf.interest_rate,
        lf.installment,
        lf.grade_numeric,
        lf.term_months,

        -- Borrower demographics
        lf.annual_income,
        lf.employment_years,
        lf.is_homeowner,
        lf.is_income_verified,

        -- Credit scores
        lf.fico_score_avg,
        lf.fico_range_low,
        lf.fico_range_high,

        -- Debt metrics
        lf.debt_to_income,
        lf.revolving_balance,
        lf.revolving_utilization,
        lf.total_balance_ex_mortgage,

        -- Derived ratios
        lf.loan_to_income_ratio,
        lf.payment_burden_ratio,
        lf.credit_history_years,
        lf.account_open_ratio,

        -- Delinquency history
        lf.delinquencies_2yrs,
        lf.months_since_last_delinquency,
        lf.accounts_now_delinquent,
        lf.num_accounts_ever_120days_past_due,
        lf.num_tradelines_30days_past_due,
        lf.num_tradelines_90days_past_due_24months,
        lf.pct_tradelines_never_delinquent,

        -- Credit inquiries
        lf.inquiries_last_6months,
        lf.inquiries_last_12months,
        lf.inquiries_financial_institutions,

        -- Account information
        lf.open_accounts,
        lf.total_accounts,
        lf.mortgage_accounts,
        lf.accounts_opened_past_24months,
        lf.num_tradelines_opened_past_12months,

        -- Public records
        lf.public_records,
        lf.public_record_bankruptcies,
        lf.tax_liens,

        -- Installment metrics
        lf.installment_utilization,
        lf.total_balance_installment,
        lf.open_installment_12months,
        lf.open_installment_24months,

        -- Bank card metrics
        lf.bank_card_utilization,
        lf.num_bank_card_tradelines,
        lf.num_active_bank_card_tradelines,
        lf.total_bank_card_limit,
        lf.percent_bank_card_gt_75,

        -- Collections
        lf.collections_12months_ex_medical,
        lf.total_collection_amount,

        -- Loan purpose
        lf.is_debt_consolidation,
        lf.loan_purpose,

        -- Hardship flags
        coalesce(lf.hardship_flag, 'N') as hardship_flag,
        coalesce(lf.settlement_status, 'N') as settlement_status,

        -- Customer history features (if available)
        coalesce(ch.total_loans, 1) as customer_total_loans,
        coalesce(ch.total_defaults, 0) as customer_total_defaults,
        coalesce(ch.historical_default_rate, 0) as customer_historical_default_rate,
        coalesce(ch.total_borrowed, lf.loan_amount) as customer_total_borrowed,
        coalesce(ch.avg_loan_amount, lf.loan_amount) as customer_avg_loan_amount,
        coalesce(ch.customer_tenure_months, 0) as customer_tenure_months,

        -- Date information (for time-based splits)
        lf.issue_date,
        lf.loan_status

    from loan_features lf
    left join customer_history ch
        on lf.member_id = ch.member_id

    where lf.is_default is not null  -- Only include completed loans
)

select * from final_dataset
