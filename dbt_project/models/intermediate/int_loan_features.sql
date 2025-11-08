{{
  config(
    materialized='view'
  )
}}

with loans as (
    select * from {{ ref('stg_loans') }}
),

loan_features as (
    select
        loan_id,

        -- Original loan fields
        loan_amount,
        funded_amount,
        interest_rate,
        installment,
        grade,
        sub_grade,
        term,
        loan_purpose,

        -- Borrower info
        annual_income,
        employment_length,
        home_ownership,
        verification_status,

        -- Credit scores
        fico_score_avg,
        fico_range_low,
        fico_range_high,

        -- Debt metrics
        debt_to_income,
        revolving_balance,
        revolving_utilization,
        total_balance_ex_mortgage,

        -- Credit history length
        earliest_credit_line,

        -- Calculate credit age in years (approximate)
        case
            when earliest_credit_line is not null
            then (julianday(issue_date) - julianday(earliest_credit_line)) / 365.25
            else null
        end as credit_history_years,

        -- Delinquencies
        delinquencies_2yrs,
        months_since_last_delinquency,
        accounts_now_delinquent,

        -- Inquiries (credit seeking behavior)
        inquiries_last_6months,
        inquiries_last_12months,
        inquiries_financial_institutions,

        -- Account counts
        open_accounts,
        total_accounts,
        mortgage_accounts,

        -- Account utilization
        case
            when total_accounts > 0
            then cast(open_accounts as float) / total_accounts
            else 0
        end as account_open_ratio,

        -- Public records
        public_records,
        public_record_bankruptcies,
        tax_liens,

        -- Loan to income ratio
        case
            when annual_income > 0
            then loan_amount / annual_income
            else null
        end as loan_to_income_ratio,

        -- Payment burden (monthly payment as % of monthly income)
        case
            when annual_income > 0
            then (installment * 12) / annual_income
            else null
        end as payment_burden_ratio,

        -- Grade to numeric conversion for modeling
        case grade
            when 'A' then 1
            when 'B' then 2
            when 'C' then 3
            when 'D' then 4
            when 'E' then 5
            when 'F' then 6
            when 'G' then 7
            else null
        end as grade_numeric,

        -- Term to numeric (months)
        case
            when term like '%36%' then 36
            when term like '%60%' then 60
            else null
        end as term_months,

        -- Employment length to numeric (years)
        case
            when employment_length = '< 1 year' then 0
            when employment_length = '1 year' then 1
            when employment_length = '2 years' then 2
            when employment_length = '3 years' then 3
            when employment_length = '4 years' then 4
            when employment_length = '5 years' then 5
            when employment_length = '6 years' then 6
            when employment_length = '7 years' then 7
            when employment_length = '8 years' then 8
            when employment_length = '9 years' then 9
            when employment_length = '10+ years' then 10
            else null
        end as employment_years,

        -- Home ownership to binary
        case
            when home_ownership in ('OWN', 'MORTGAGE') then 1
            else 0
        end as is_homeowner,

        -- Income verification
        case
            when verification_status in ('Verified', 'Source Verified') then 1
            else 0
        end as is_income_verified,

        -- High risk loan purposes
        case
            when loan_purpose in ('debt_consolidation', 'credit_card') then 1
            else 0
        end as is_debt_consolidation,

        -- Installment utilization metrics
        installment_utilization,
        total_balance_installment,
        open_installment_12months,
        open_installment_24months,

        -- Bank card metrics
        bank_card_utilization,
        num_bank_card_tradelines,
        num_active_bank_card_tradelines,
        total_bank_card_limit,
        max_balance_bank_card,
        percent_bank_card_gt_75,

        -- Delinquency indicators
        num_accounts_ever_120days_past_due,
        num_tradelines_30days_past_due,
        num_tradelines_90days_past_due_24months,
        pct_tradelines_never_delinquent,

        -- Recent account opening activity
        accounts_opened_past_24months,
        num_tradelines_opened_past_12months,
        open_accounts_6months,

        -- Collections
        collections_12months_ex_medical,
        total_collection_amount,

        -- Hardship indicators
        hardship_flag,
        settlement_status,

        -- Target variable
        is_default,
        loan_status,
        issue_date

    from loans
)

select * from loan_features
