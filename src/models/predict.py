#!/usr/bin/env python3
"""
Prediction Script for Loan Default Prediction

This script loads a trained model and makes predictions on new loan data.
"""

import sys
from pathlib import Path
import pickle
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class LoanDefaultPredictor:
    """Handles predictions using trained loan default model"""

    def __init__(self, model_dir: str = "outputs/models"):
        """
        Initialize the predictor

        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_names = None
        self.metadata = None

    def load_model(self, model_name: str = None):
        """
        Load a trained model and its artifacts

        Args:
            model_name: Name of the model to load (if None, loads best from metadata)
        """
        # Load metadata to get best model
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                if model_name is None:
                    model_name = self.metadata['model_name']

        if model_name is None:
            raise ValueError("No model name provided and no metadata found")

        print(f"Loading model: {model_name}")

        # Load model
        model_path = self.model_dir / f"{model_name}_model.pkl"
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load scaler
        scaler_path = self.model_dir / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load encoders
        encoders_path = self.model_dir / "encoders.pkl"
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)

        # Load feature names
        feature_names_path = self.model_dir / "feature_names.json"
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)

        print(f"✓ Model loaded successfully")
        print(f"  Model: {model_name}")
        print(f"  Features: {len(self.feature_names)}")
        if self.metadata:
            print(f"  AUC: {self.metadata['metrics']['auc']:.3f}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction

        Args:
            df: Input dataframe with loan data

        Returns:
            Processed feature dataframe
        """
        # Ensure all required features are present
        # Fill missing features with 0 or median
        X = pd.DataFrame()

        for feature in self.feature_names:
            if feature in df.columns:
                X[feature] = df[feature]
            else:
                # Feature not in input - use default value
                X[feature] = 0

        # Fill missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
            index=X.index
        )

        return X_scaled

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data

        Args:
            df: Input dataframe with loan data

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Prepare features
        X = self.prepare_features(df)

        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        # Create results dataframe
        results = pd.DataFrame({
            'prediction': predictions,
            'default_probability': probabilities,
            'risk_level': pd.cut(
                probabilities,
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        }, index=df.index)

        return results

    def predict_single(self, loan_data: dict) -> dict:
        """
        Make prediction for a single loan

        Args:
            loan_data: Dictionary with loan features

        Returns:
            Dictionary with prediction results
        """
        # Convert to dataframe
        df = pd.DataFrame([loan_data])

        # Get prediction
        result = self.predict(df)

        return {
            'prediction': int(result['prediction'].iloc[0]),
            'default_probability': float(result['default_probability'].iloc[0]),
            'risk_level': str(result['risk_level'].iloc[0]),
            'recommendation': self._get_recommendation(
                result['default_probability'].iloc[0]
            )
        }

    def _get_recommendation(self, probability: float) -> str:
        """
        Get recommendation based on default probability

        Args:
            probability: Default probability

        Returns:
            Recommendation string
        """
        if probability < 0.2:
            return "APPROVE - Low risk"
        elif probability < 0.4:
            return "APPROVE - Acceptable risk"
        elif probability < 0.6:
            return "REVIEW - Moderate risk, consider higher interest rate"
        elif probability < 0.8:
            return "REVIEW - High risk, recommend manual review"
        else:
            return "REJECT - Very high risk"


def main():
    """Main execution function"""
    print("=" * 60)
    print("Loan Default Prediction - Predictor")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "outputs" / "models"

    # Check if model exists
    if not model_dir.exists():
        print(f"\n❌ Model directory not found: {model_dir}")
        print("\nPlease train a model first:")
        print("  python src/models/train.py")
        sys.exit(1)

    # Initialize predictor
    predictor = LoanDefaultPredictor(model_dir=str(model_dir))

    # Load model
    predictor.load_model()

    # Example usage
    print("\n" + "=" * 60)
    print("Example Usage:")
    print("=" * 60)

    example_loan = {
        'loan_amount': 10000,
        'interest_rate': 10.5,
        'installment': 325.50,
        'grade_numeric': 2,  # Grade B
        'term_months': 36,
        'annual_income': 65000,
        'employment_years': 5,
        'is_homeowner': 1,
        'is_income_verified': 1,
        'fico_score_avg': 700,
        'debt_to_income': 18.5,
        'revolving_balance': 12000,
        'revolving_utilization': 45.5,
        'delinquencies_2yrs': 0,
        'inquiries_last_6months': 1,
        'open_accounts': 10,
        'total_accounts': 15,
        'public_records': 0,
        'loan_to_income_ratio': 0.154,
        'payment_burden_ratio': 0.06
    }

    print("\nExample loan application:")
    print(f"  Loan amount: ${example_loan['loan_amount']:,}")
    print(f"  Annual income: ${example_loan['annual_income']:,}")
    print(f"  FICO score: {example_loan['fico_score_avg']}")
    print(f"  DTI: {example_loan['debt_to_income']}%")

    result = predictor.predict_single(example_loan)

    print("\nPrediction:")
    print(f"  Default probability: {result['default_probability']:.1%}")
    print(f"  Risk level: {result['risk_level']}")
    print(f"  Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()
