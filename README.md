# Loan Default Prediction System

A comprehensive end-to-end machine learning system for predicting loan defaults in personal lending, featuring SQL data extraction, dbt transformations, and Python ML models.

## Project Overview

This project demonstrates a production-ready approach to loan default prediction, incorporating:
- **SQL**: Data extraction and initial transformations
- **dbt**: Modular data transformations, testing, and documentation
- **Python**: Feature engineering, ML modeling, and evaluation
- **MLOps**: Model versioning, monitoring, and deployment patterns

## Business Context

In personal lending (small loans and payday loans), predicting default risk is critical for:
- Setting appropriate interest rates and credit limits
- Optimizing approval decisions to balance risk and growth
- Early identification of at-risk accounts for proactive collection
- Portfolio monitoring and regulatory compliance

## Architecture

```
┌─────────────────┐
│  Raw Data       │
│  (Lending Club  │
│   Dataset)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SQL Layer      │
│  - Data Extract │
│  - Basic Joins  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  dbt Layer      │
│  - Staging      │
│  - Features     │
│  - Marts        │
│  - Tests        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Layer       │
│  - Feature Eng  │
│  - Training     │
│  - Evaluation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deployment     │
│  - API          │
│  - Monitoring   │
└─────────────────┘
```

## Dataset

We'll use the Lending Club dataset, which contains real loan data with features like:
- Loan amount, term, interest rate
- Borrower credit score, income, employment length
- Debt-to-income ratio, delinquency history
- Loan purpose, home ownership status
- Loan status (fully paid, charged off, current, etc.)

**Download**: [Kaggle Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## Project Structure

```
loan_default_prediction/
├── README.md
├── data/
│   ├── raw/                    # Original Lending Club CSV files
│   ├── processed/              # Cleaned datasets
│   └── features/               # Feature engineered datasets
├── dbt_project/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/            # Initial data cleaning
│   │   ├── intermediate/       # Feature calculations
│   │   └── marts/              # Business-ready datasets
│   ├── macros/                 # Reusable dbt macros
│   ├── tests/                  # Custom data tests
│   └── seeds/                  # Reference data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py     # Data loading utilities
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py           # Model training scripts
│   │   ├── evaluate.py        # Model evaluation
│   │   └── predict.py         # Prediction pipeline
│   └── utils/
│       ├── config.py          # Configuration management
│       └── logger.py          # Logging utilities
├── config/
│   ├── model_config.yaml      # ML model configurations
│   └── features.yaml          # Feature definitions
├── outputs/
│   ├── models/                # Trained model artifacts
│   ├── metrics/               # Model performance metrics
│   └── reports/               # Generated reports
└── requirements.txt
```

## Key Features to Build

### Behavioral Features
- **Payment history patterns**: Days delinquent, missed payments, payment consistency
- **Loan utilization**: Current balance vs original amount, paydown velocity
- **Customer tenure**: Months since first loan, number of previous loans

### Financial Features
- **Debt-to-income calculations**: Including revolving debt, installment debt
- **Credit utilization**: Across all credit lines
- **Income stability**: Employment length, income verification status

### Risk Indicators
- **Delinquency flags**: Recent bankruptcies, tax liens, charge-offs
- **Credit inquiries**: Recent hard pulls indicating credit-seeking behavior
- **Loan purpose risk scores**: Different default rates by loan purpose

### Aggregated Features
- **Rolling statistics**: 3/6/12 month payment averages
- **Cohort comparisons**: Performance vs similar loans in same vintage
- **Seasonal patterns**: Application/default timing patterns

## dbt Models Structure

### Staging Layer
- `stg_loans`: Clean raw loan data, standardize datatypes
- `stg_payments`: Payment history records
- `stg_credit_bureau`: Credit report data

### Intermediate Layer
- `int_loan_features`: Calculated loan-level features (utilization, age, etc.)
- `int_customer_history`: Customer aggregation (total borrowed, repayment rate)
- `int_payment_patterns`: Rolling payment metrics

### Mart Layer
- `fct_loans`: Fact table with all loans and features
- `dim_customers`: Customer dimension with aggregated history
- `mart_ml_training`: Final dataset for ML training with target variable

## ML Models to Implement

1. **Logistic Regression**: Baseline, interpretable model
2. **Random Forest**: Capture non-linear relationships
3. **Gradient Boosting** (XGBoost/LightGBM): Best performance
4. **Neural Network**: For comparison and learning

## Evaluation Metrics

- **AUC-ROC**: Overall discrimination ability
- **Precision/Recall**: At different threshold levels
- **Gini Coefficient**: Common in credit risk
- **KS Statistic**: Maximum separation between good/bad loans
- **Business metrics**: Expected loss, approval rate at various cutoffs

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data
- Download Lending Club dataset from Kaggle
- Place in `data/raw/` directory

### 3. Database Setup
```bash
# For local development, we'll use DuckDB (SQLite alternative)
# Or set up PostgreSQL if you prefer a production-like environment
```

### 4. dbt Setup
```bash
cd dbt_project
dbt deps
dbt seed
dbt run
dbt test
```

### 5. Run ML Pipeline
```bash
python src/models/train.py --config config/model_config.yaml
```

## Next Steps

1. **Data Acquisition**: Download Lending Club dataset
2. **Database Setup**: Initialize DuckDB or PostgreSQL
3. **dbt Models**: Build staging → intermediate → marts
4. **Feature Engineering**: Create domain-specific features
5. **Model Training**: Implement multiple algorithms
6. **Model Evaluation**: Compare performance and business impact
7. **API Development**: Create prediction endpoint
8. **Monitoring**: Track model performance over time

## Advanced Extensions

- **Temporal validation**: Train on older data, validate on newer vintages
- **Fairness analysis**: Check for disparate impact across demographics
- **Explainability**: SHAP values for model interpretability
- **A/B testing framework**: Compare model versions in production
- **Auto-retraining pipeline**: Scheduled model updates
- **Feature store**: Centralized feature management
- **Model registry**: MLflow or similar for versioning

## Learning Outcomes

By completing this project, you'll demonstrate:
- ✅ Advanced SQL for data extraction and joins
- ✅ dbt best practices (modularity, testing, documentation)
- ✅ Python ML pipeline development
- ✅ Feature engineering for credit risk
- ✅ Model evaluation and comparison
- ✅ Domain expertise in lending analytics
- ✅ End-to-end ML system design
