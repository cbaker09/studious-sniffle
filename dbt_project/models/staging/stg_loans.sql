{{
  config(
    materialized='view'
  )
}}

with source_data as (
    select * from loans
),

cleaned as (
    select
        -- Primary key
        id as loan_id,
        member_id,

        -- Loan details
        loan_amnt as loan_amount,
        funded_amnt as funded_amount,
        funded_amnt_inv as funded_amount_investor,
        term,
        int_rate as interest_rate,
        installment,
        grade,
        sub_grade,

        -- Dates
        issue_d as issue_date,

        -- Loan status
        loan_status,

        -- Payment info
        pymnt_plan as payment_plan,

        -- Purpose
        purpose as loan_purpose,
        title as loan_title,

        -- Geographic
        zip_code,
        addr_state as state,

        -- Borrower info
        emp_title as employer_title,
        emp_length as employment_length,
        home_ownership,
        annual_inc as annual_income,
        verification_status,

        -- Credit history
        dti as debt_to_income,
        delinq_2yrs as delinquencies_2yrs,
        earliest_cr_line as earliest_credit_line,
        fico_range_low,
        fico_range_high,
        inq_last_6mths as inquiries_last_6months,
        mths_since_last_delinq as months_since_last_delinquency,
        mths_since_last_record as months_since_last_public_record,
        open_acc as open_accounts,
        pub_rec as public_records,
        revol_bal as revolving_balance,
        revol_util as revolving_utilization,
        total_acc as total_accounts,

        -- Collections
        collections_12_mths_ex_med as collections_12months_ex_medical,

        -- Policy code
        policy_code,

        -- Application type
        application_type,

        -- Joint info
        annual_inc_joint as annual_income_joint,
        dti_joint as debt_to_income_joint,
        verification_status_joint,

        -- Account info
        acc_now_delinq as accounts_now_delinquent,
        tot_coll_amt as total_collection_amount,
        tot_cur_bal as total_current_balance,
        open_acc_6m as open_accounts_6months,
        open_act_il as open_active_installment,
        open_il_12m as open_installment_12months,
        open_il_24m as open_installment_24months,
        mths_since_rcnt_il as months_since_recent_installment,
        total_bal_il as total_balance_installment,
        il_util as installment_utilization,
        open_rv_12m as open_revolving_12months,
        open_rv_24m as open_revolving_24months,
        max_bal_bc as max_balance_bank_card,
        all_util as all_utilization,
        total_rev_hi_lim as total_revolving_high_limit,
        inq_fi as inquiries_financial_institutions,
        total_cu_tl as total_credit_union_tradelines,
        inq_last_12m as inquiries_last_12months,

        -- Additional credit metrics
        acc_open_past_24mths as accounts_opened_past_24months,
        avg_cur_bal as average_current_balance,
        bc_open_to_buy as bank_card_open_to_buy,
        bc_util as bank_card_utilization,
        chargeoff_within_12_mths as chargeoffs_within_12months,
        delinq_amnt as delinquency_amount,
        mo_sin_old_il_acct as months_since_oldest_installment,
        mo_sin_old_rev_tl_op as months_since_oldest_revolving,
        mo_sin_rcnt_rev_tl_op as months_since_recent_revolving,
        mo_sin_rcnt_tl as months_since_recent_tradeline,
        mort_acc as mortgage_accounts,
        mths_since_recent_bc as months_since_recent_bank_card,
        mths_since_recent_bc_dlq as months_since_recent_bank_card_delinquency,
        mths_since_recent_inq as months_since_recent_inquiry,
        mths_since_recent_revol_delinq as months_since_recent_revolving_delinquency,
        num_accts_ever_120_pd as num_accounts_ever_120days_past_due,
        num_actv_bc_tl as num_active_bank_card_tradelines,
        num_actv_rev_tl as num_active_revolving_tradelines,
        num_bc_sats as num_bank_card_satisfactory,
        num_bc_tl as num_bank_card_tradelines,
        num_il_tl as num_installment_tradelines,
        num_op_rev_tl as num_open_revolving_tradelines,
        num_rev_accts as num_revolving_accounts,
        num_rev_tl_bal_gt_0 as num_revolving_tradelines_balance_gt_0,
        num_sats as num_satisfactory_accounts,
        num_tl_120dpd_2m as num_tradelines_120days_past_due_2months,
        num_tl_30dpd as num_tradelines_30days_past_due,
        num_tl_90g_dpd_24m as num_tradelines_90days_past_due_24months,
        num_tl_op_past_12m as num_tradelines_opened_past_12months,
        pct_tl_nvr_dlq as pct_tradelines_never_delinquent,
        percent_bc_gt_75 as percent_bank_card_gt_75,
        pub_rec_bankruptcies as public_record_bankruptcies,
        tax_liens,
        tot_hi_cred_lim as total_high_credit_limit,
        total_bal_ex_mort as total_balance_ex_mortgage,
        total_bc_limit as total_bank_card_limit,
        total_il_high_credit_limit,

        -- Hardship flags
        hardship_flag,
        hardship_type,
        hardship_reason,
        hardship_status,
        deferral_term,
        hardship_amount,
        hardship_start_date,
        hardship_end_date,
        payment_plan_start_date,
        hardship_length,
        hardship_dpd as hardship_days_past_due,
        hardship_loan_status,
        orig_projected_additional_accrued_interest,
        hardship_payoff_balance_amount,
        hardship_last_payment_amount,

        -- Settlement info
        settlement_status,
        settlement_date,
        settlement_amount,
        settlement_percentage,
        settlement_term,

        -- Calculated fields
        case
            when loan_status in {{ var('default_statuses') | join(', ', attribute='|tojson') }}
            then 1
            when loan_status in {{ var('paid_statuses') | join(', ', attribute='|tojson') }}
            then 0
            else null
        end as is_default,

        -- FICO average
        (fico_range_low + fico_range_high) / 2.0 as fico_score_avg,

        -- Current timestamp
        current_timestamp as loaded_at

    from source_data
)

select * from cleaned
where is_default is not null  -- Only include completed loans (paid or defaulted)
