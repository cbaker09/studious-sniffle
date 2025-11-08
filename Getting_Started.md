# Getting Started Guide: Loan Default Prediction System

This guide will walk you through setting up and running your first loan default prediction model.

## Prerequisites

- Python 3.9 or higher
- 4GB+ RAM recommended
- 2GB disk space for data and models

## Step-by-Step Setup

### Step 1: Download the Lending Club Dataset

1. Visit Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Download the dataset (you'll need a Kaggle account)
3. Extract the CSV file(s)
4. Place in `data/raw/` directory

Example file structure:
```
loan_default_prediction/
└── data/
    └── raw/
        └── loan.csv  (or accepted_2007_to_2018.csv)
```

### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 3: Load Data into DuckDB

```bash
# Run the data loader
python src/data/data_loader.py
```

This will:
- Create a DuckDB database at `data/loan_default.duckdb`
- Load your CSV data into a `loans` table
- Display validation statistics
- Create a sample dataset for testing

**Example output:**
```
Loading CSV from data/raw/loan.csv
Target table: loans
Loaded chunk 1: 50000 rows (Total: 50000)
Loaded chunk 2: 50000 rows (Total: 100000)
...
Successfully loaded 245,000 total rows into loans

=== Data Validation Results ===
Total rows: 245,000
Default count: 37,450
Default rate: 15.3%
```

### Step 4: Run dbt Models

```bash
# Navigate to dbt project
cd dbt_project

# Install dbt packages
dbt deps

# Run all models
dbt run

# Run tests
dbt test

# Generate documentation
dbt docs generate
dbt docs serve
```

**What this does:**
1. **Staging layer** (`stg_loans`): Cleans raw data, standardizes columns
2. **Intermediate layer** (`int_loan_features`): Calculates derived features
3. **Marts layer** (`mart_ml_training`): Creates final ML-ready dataset

**Example output:**
```
Running with dbt=1.7.4
Found 3 models, 12 tests, 0 snapshots

Completed successfully

Done. PASS=3 WARN=0 ERROR=0 SKIP=0 TOTAL=3
```

### Step 5: Train Models

```bash
# Return to project root
cd ..

# Run training pipeline
python src/models/train.py
```

This will train 4 different models:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. LightGBM

**Example output:**
```
=== Loan Default Prediction - Model Training ===

Loaded 245,000 loans with 15.3% default rate
Training set: 196,000 samples
Test set: 49,000 samples

=== Training Logistic Regression ===
✓ Logistic Regression trained

=== Training Random Forest ===
✓ Random Forest trained

=== Training XGBoost ===
✓ XGBoost trained

=== Training LightGBM ===
✓ LightGBM trained

=== Final Model Comparison ===
         Model    AUC   Gini  Precision  Recall     F1
      xgboost  0.703  0.406      0.485   0.682  0.567
      lightgbm 0.698  0.396      0.478   0.671  0.558
 random_forest 0.689  0.378      0.463   0.655  0.542
logistic_regression 0.672  0.344  0.441   0.628  0.518

✓ Saved best model: xgboost
```

### Step 6: Explore Results

After training, you'll find:

```
outputs/
├── models/
│   ├── xgboost_model.pkl      # Best model
│   ├── scaler.pkl              # Feature scaler
│   ├── encoders.pkl            # Categorical encoders
│   └── feature_names.json      # Feature list
├── metrics/
│   └── model_comparison.csv    # Performance metrics
└── reports/
    └── (future: detailed reports)
```

## Understanding the Results

### Model Performance Metrics

**AUC (Area Under ROC Curve)**: 0.70 means the model has 70% chance of ranking a random defaulted loan higher than a random non-defaulted loan. In lending:
- 0.65-0.70: Acceptable
- 0.70-0.75: Good
- 0.75+: Excellent

**Gini Coefficient**: Alternative to AUC, calculated as `2*AUC - 1`. Higher is better (0-1 range).

**Precision**: Of loans predicted to default, what % actually defaulted?
- High precision = fewer false alarms
- Important for not rejecting good customers

**Recall**: Of all loans that defaulted, what % did we catch?
- High recall = catching more bad loans
- Important for minimizing losses

**F1 Score**: Harmonic mean of precision and recall
- Balances both metrics

### Business Impact

With a model AUC of 0.70:
- You can improve approval decisions
- Better price loans based on risk
- Identify high-risk accounts early
- Reduce portfolio default rate by 10-20%

Example: If your current default rate is 15%, a good model could reduce it to 12-13%, saving millions in losses.

## Next Steps

### 1. Jupyter Notebook Exploration

Create a notebook to explore results:

```python
import pandas as pd
import pickle

# Load model
with open('outputs/models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Check feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

### 2. Hyperparameter Tuning

Improve model performance:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=3,
    scoring='roc_auc'
)
```

### 3. Feature Engineering

Add new features in dbt:
- Payment velocity (how fast customers pay down loans)
- Seasonal patterns (default rates by month)
- Cohort comparisons (performance vs similar loans)
- External data (unemployment rates, interest rate environment)

### 4. Model Monitoring

Track model performance over time:
- Default rate predictions vs actuals
- Feature drift (are features changing?)
- Population stability index (PSI)
- Model recalibration schedule

### 5. Deployment

Build a prediction API:

```python
from fastapi import FastAPI
import pickle

app = FastAPI()

@app.post("/predict")
def predict_default(loan_features: dict):
    # Load model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make prediction
    probability = model.predict_proba([features])[0][1]
    
    return {"default_probability": probability}
```

## Common Issues & Solutions

### Issue: "CSV file not found"
**Solution**: Update the path in `src/data/data_loader.py` to point to your downloaded CSV

### Issue: "Out of memory"
**Solution**: Use the sample dataset creation feature or reduce chunk size

### Issue: "dbt can't find models"
**Solution**: Ensure you're in the `dbt_project` directory when running dbt commands

### Issue: Low model performance
**Solutions**:
- Check for data quality issues (run `dbt test`)
- Try hyperparameter tuning
- Add more features
- Use more training data
- Check for data leakage

## Tips for Portfolio Projects

1. **Document everything**: Add comments explaining your decisions
2. **Version control**: Use git to track changes
3. **Create visualizations**: Charts showing feature importance, ROC curves, etc.
4. **Write a blog post**: Explain your approach and findings
5. **Deploy a demo**: Even a simple Streamlit app shows production skills

## Resources

- **dbt Documentation**: https://docs.getdbt.com
- **XGBoost Guide**: https://xgboost.readthedocs.io
- **Credit Risk Modeling**: "Credit Risk Analytics" by Bart Baesens
- **Model Evaluation**: https://scikit-learn.org/stable/modules/model_evaluation.html

## Questions?

This is a learning project! Common questions:

**Q: What's a good AUC score for credit risk?**
A: 0.65-0.70 is typical for personal loans, 0.70+ is good

**Q: Should I use random or temporal split?**
A: Temporal split better mimics production (train on past, predict future)

**Q: How often should I retrain?**
A: Quarterly is common, monthly if you have rapid portfolio changes

**Q: What features matter most?**
A: Typically: credit score, DTI, payment history, income verification

Happy modeling! 🎯
