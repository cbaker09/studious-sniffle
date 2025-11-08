#!/usr/bin/env python3
"""
Model Evaluation Script for Loan Default Prediction

This script provides detailed evaluation and visualization of trained models.
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class ModelEvaluator:
    """Handles detailed evaluation of trained models"""

    def __init__(self, model_dir: str = "outputs/models"):
        """
        Initialize the evaluator

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
        print(f"  Features: {len(self.feature_names)}")

    def plot_roc_curve(self, y_true, y_pred_proba, save_path: str = None):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path: str = None):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Precision-Recall curve saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """Plot feature importance"""
        # Get feature importance (works for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Plot top N
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(top_n)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances')
            plt.gca().invert_yaxis()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Feature importance plot saved to {save_path}")
            else:
                plt.show()

            plt.close()

            return feature_importance

        else:
            print("Feature importance not available for this model")
            return None

    def plot_confusion_matrix(self, y_true, y_pred, save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def evaluate(self, X_test, y_test, output_dir: str = None):
        """
        Run comprehensive evaluation

        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save plots (if None, displays plots)
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Make predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        # Print classification report
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))

        # Plot ROC curve
        roc_path = output_dir / "roc_curve.png" if output_dir else None
        self.plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)

        # Plot Precision-Recall curve
        pr_path = output_dir / "precision_recall_curve.png" if output_dir else None
        self.plot_precision_recall_curve(y_test, y_pred_proba, save_path=pr_path)

        # Plot confusion matrix
        cm_path = output_dir / "confusion_matrix.png" if output_dir else None
        self.plot_confusion_matrix(y_test, y_pred, save_path=cm_path)

        # Plot feature importance
        fi_path = output_dir / "feature_importance.png" if output_dir else None
        feature_importance = self.plot_feature_importance(save_path=fi_path)

        if feature_importance is not None and output_dir:
            fi_csv_path = output_dir / "feature_importance.csv"
            feature_importance.to_csv(fi_csv_path, index=False)
            print(f"✓ Feature importance saved to {fi_csv_path}")


def main():
    """Main execution function"""
    print("=" * 60)
    print("Loan Default Prediction - Model Evaluation")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "outputs" / "models"
    reports_dir = project_root / "outputs" / "reports"

    # Check if model exists
    if not model_dir.exists():
        print(f"\n❌ Model directory not found: {model_dir}")
        print("\nPlease train a model first:")
        print("  python src/models/train.py")
        sys.exit(1)

    # Initialize evaluator
    evaluator = ModelEvaluator(model_dir=str(model_dir))

    # Load model
    evaluator.load_model()

    print("\nNote: For full evaluation, load your test data and call evaluator.evaluate(X_test, y_test)")
    print("Example:")
    print("  evaluator.evaluate(X_test, y_test, output_dir='outputs/reports')")


if __name__ == "__main__":
    main()
