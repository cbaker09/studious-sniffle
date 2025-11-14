#!/usr/bin/env python3
"""
Model Training Script for Loan Default Prediction

This script trains multiple ML models on the loan default dataset
and saves the best performing model.
"""

import os
import sys
from pathlib import Path
import pickle
import json
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()


class LoanDefaultModelTrainer:
    """Handles training multiple models for loan default prediction"""

    def __init__(
        self,
        database_url: str = None,
        output_dir: str = "outputs"
    ):
        """
        Initialize the model trainer

        Args:
            database_url: PostgreSQL database URL (defaults to DATABASE_URL env var)
            output_dir: Directory to save models and metrics
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'postgresql://loan_user:loan_password@localhost:5432/loan_default'
        )
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.metrics_dir = self.output_dir / "metrics"

        # Create output directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.encoders = {}

        # Models
        self.models = {}
        self.results = {}

    def load_data(self, table_name: str = "mart_ml_training", schema: str = "marts"):
        """
        Load training data from PostgreSQL

        Args:
            table_name: Name of the table containing ML training data
            schema: Database schema (default: 'marts')
        """
        print(f"Loading data from {schema}.{table_name}...")

        # Create database engine
        engine = create_engine(self.database_url)

        # Load data
        df = pd.read_sql(f"SELECT * FROM {schema}.{table_name}", engine)
        engine.dispose()

        print(f"✓ Loaded {len(df):,} loans")

        # Check default rate
        default_rate = df['is_default'].mean()
        print(f"Default rate: {default_rate:.1%}")

        return df

    def prepare_features(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Prepare features for modeling

        Args:
            df: Input dataframe
            test_size: Proportion of data to use for testing
        """
        print("\nPreparing features...")

        # Separate target from features
        target = 'is_default'
        exclude_cols = [
            'loan_id',
            'is_default',
            'loan_status',
            'issue_date',
            'loan_purpose'  # Will encode this
        ]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Encode categorical features
        categorical_features = [
            'hardship_flag',
            'settlement_status'
        ]

        df_encoded = df.copy()

        for col in categorical_features:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le

        # Encode loan purpose (one-hot)
        if 'loan_purpose' in df.columns:
            purpose_dummies = pd.get_dummies(
                df['loan_purpose'],
                prefix='purpose',
                drop_first=True
            )
            df_encoded = pd.concat([df_encoded, purpose_dummies], axis=1)
            feature_cols.extend(purpose_dummies.columns.tolist())

        # Get features
        X = df_encoded[feature_cols].copy()
        y = df_encoded[target].copy()

        # Fill missing values with median for numeric columns
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
            else:
                X[col] = X[col].fillna(0)

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Time-based split (more realistic for deployment)
        # Use issue_date to split - train on older loans, test on newer
        if 'issue_date' in df.columns:
            print("Using time-based split (train on older, test on newer loans)...")
            df_sorted = df.sort_values('issue_date')
            split_idx = int(len(df_sorted) * (1 - test_size))

            train_idx = df_sorted.index[:split_idx]
            test_idx = df_sorted.index[split_idx:]

            self.X_train = X.loc[train_idx]
            self.X_test = X.loc[test_idx]
            self.y_train = y.loc[train_idx]
            self.y_test = y.loc[test_idx]
        else:
            # Random split
            print("Using random split...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        # Scale features
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index
        )

        print(f"✓ Training set: {len(self.X_train):,} samples")
        print(f"✓ Test set: {len(self.X_test):,} samples")
        print(f"✓ Features: {len(self.feature_names)}")

    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n=== Training Logistic Regression ===")

        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )

        model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = model

        print("✓ Logistic Regression trained")

    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n=== Training Random Forest ===")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model

        print("✓ Random Forest trained")

    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n=== Training XGBoost ===")

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        )

        model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = model

        print("✓ XGBoost trained")

    def train_lightgbm(self):
        """Train LightGBM model"""
        print("\n=== Training LightGBM ===")

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(self.X_train, self.y_train)
        self.models['lightgbm'] = model

        print("✓ LightGBM trained")

    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n=== Evaluating Models ===")

        results_list = []

        for name, model in self.models.items():
            # Predictions
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = model.predict(self.X_test)

            # Calculate metrics
            auc = roc_auc_score(self.y_test, y_pred_proba)
            gini = 2 * auc - 1  # Gini coefficient
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            results_list.append({
                'model': name,
                'auc': auc,
                'gini': gini,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

            self.results[name] = {
                'auc': auc,
                'gini': gini,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

        # Create comparison dataframe
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('auc', ascending=False)

        print("\n=== Model Comparison ===")
        print(results_df.to_string(index=False))

        # Save results
        results_path = self.metrics_dir / "model_comparison.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path}")

        return results_df

    def save_best_model(self, results_df: pd.DataFrame):
        """
        Save the best performing model

        Args:
            results_df: DataFrame with model comparison results
        """
        # Get best model by AUC
        best_model_name = results_df.iloc[0]['model']
        best_model = self.models[best_model_name]

        print(f"\n=== Saving Best Model: {best_model_name} ===")

        # Save model
        model_path = self.models_dir / f"{best_model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✓ Model saved to {model_path}")

        # Save scaler
        scaler_path = self.models_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {scaler_path}")

        # Save encoders
        encoders_path = self.models_dir / "encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.encoders, f)
        print(f"✓ Encoders saved to {encoders_path}")

        # Save feature names
        feature_names_path = self.models_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"✓ Feature names saved to {feature_names_path}")

        # Save metadata
        metadata = {
            'model_name': best_model_name,
            'trained_at': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test),
            'metrics': {
                'auc': float(results_df.iloc[0]['auc']),
                'gini': float(results_df.iloc[0]['gini']),
                'precision': float(results_df.iloc[0]['precision']),
                'recall': float(results_df.iloc[0]['recall']),
                'f1': float(results_df.iloc[0]['f1'])
            }
        }

        metadata_path = self.models_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main execution function"""
    print("=" * 60)
    print("Loan Default Prediction - Model Training")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent.parent

    # Initialize trainer (uses DATABASE_URL from .env)
    trainer = LoanDefaultModelTrainer(
        output_dir=str(project_root / "outputs")
    )

    try:
        # Load data
        df = trainer.load_data()

        # Prepare features
        trainer.prepare_features(df)

        # Train models
        trainer.train_logistic_regression()
        trainer.train_random_forest()
        trainer.train_xgboost()
        trainer.train_lightgbm()

        # Evaluate models
        results_df = trainer.evaluate_models()

        # Save best model
        trainer.save_best_model(results_df)

        print("\n" + "=" * 60)
        print("✓ Training Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\nPlease ensure:")
        print("  1. PostgreSQL is running (docker-compose up -d)")
        print("  2. Data has been loaded (python src/data/data_loader.py)")
        print("  3. dbt models have been run (cd dbt_project && dbt run)")
        sys.exit(1)


if __name__ == "__main__":
    main()
