# Loan Default Prediction System

A comprehensive end-to-end machine learning system for predicting loan defaults in personal lending, featuring SQL data extraction, dbt transformations, and Python ML models.

## Project Overview

This project demonstrates a production-ready approach to loan default prediction, incorporating:
- **PostgreSQL**: Robust relational database for data storage
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

### 2. PostgreSQL Database Setup

This project uses PostgreSQL for data storage. We provide a Docker Compose configuration for easy local setup.

#### Option A: Using Docker (Recommended)

```bash
# Start PostgreSQL container
docker-compose up -d

# Verify PostgreSQL is running
docker-compose ps

# View logs if needed
docker-compose logs postgres
```

The database will be available at:
- **Host**: localhost
- **Port**: 5432
- **Database**: loan_default
- **User**: loan_user
- **Password**: loan_password

#### Option B: Local PostgreSQL Installation

If you prefer to use a local PostgreSQL installation:

1. Install PostgreSQL 15+ on your system
2. Create a database and user:
```sql
CREATE DATABASE loan_default;
CREATE USER loan_user WITH PASSWORD 'loan_password';
GRANT ALL PRIVILEGES ON DATABASE loan_default TO loan_user;
```

3. Update the `.env` file with your connection details

#### Environment Variables

The project uses a `.env` file for database configuration. A sample file is provided:

```bash
# Copy the example file
cp .env.example .env

# Edit .env if you need to change the default settings
```

Default configuration:
```
POSTGRES_DB=loan_default
POSTGRES_USER=loan_user
POSTGRES_PASSWORD=loan_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

### 3. Download Data
- Download Lending Club dataset from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Place the CSV file in `data/raw/` directory
- Expected filename: `loan.csv` or `accepted_2007_to_2018.csv`

### 4. Load Data into PostgreSQL
```bash
# Run the data loader
python src/data/data_loader.py

# This will:
# - Connect to PostgreSQL
# - Load CSV data in chunks
# - Create tables in the 'raw' schema
# - Validate the loaded data
```

### 5. dbt Setup
```bash
cd dbt_project

# Run dbt models to create staging, intermediate, and marts tables
dbt run

# Run tests to validate data quality
dbt test

# Generate documentation (optional)
dbt docs generate
dbt docs serve
```

### 6. Run ML Pipeline
```bash
# Train models
python src/models/train.py

# This will:
# - Load data from the marts.mart_ml_training table
# - Train multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM)
# - Evaluate and compare model performance
# - Save the best model to outputs/models/
```

### 7. Stopping the Database
```bash
# Stop PostgreSQL container
docker-compose down

# Stop and remove data (caution: deletes all data!)
docker-compose down -v
```

## Database Schema

The project uses a multi-schema approach for data organization:

- **raw**: Raw data loaded from CSV files
- **staging**: Cleaned and standardized data (dbt views)
- **intermediate**: Feature engineering and calculations (dbt views)
- **marts**: Business-ready datasets for ML (dbt tables)

This follows the medallion architecture pattern commonly used in modern data platforms.

## Next Steps

1. ✅ **Database Setup**: PostgreSQL with Docker
2. ✅ **Data Pipeline**: dbt models (staging → intermediate → marts)
3. ✅ **ML Models**: Multiple algorithms implemented
4. **Model Tuning**: Hyperparameter optimization
5. **API Development**: Create prediction endpoint with FastAPI
6. **Monitoring**: Track model performance over time
7. **Deployment**: Containerize and deploy to cloud

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
