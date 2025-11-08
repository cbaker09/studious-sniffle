"""
Configuration management utilities for loan default prediction system
"""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Handles loading and accessing configuration files"""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self._model_config = None
        self._features_config = None

    def load_model_config(self) -> Dict[str, Any]:
        """
        Load model configuration

        Returns:
            Dictionary with model configuration
        """
        if self._model_config is None:
            config_path = self.config_dir / "model_config.yaml"
            with open(config_path, 'r') as f:
                self._model_config = yaml.safe_load(f)

        return self._model_config

    def load_features_config(self) -> Dict[str, Any]:
        """
        Load features configuration

        Returns:
            Dictionary with features configuration
        """
        if self._features_config is None:
            config_path = self.config_dir / "features.yaml"
            with open(config_path, 'r') as f:
                self._features_config = yaml.safe_load(f)

        return self._features_config

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model parameters
        """
        config = self.load_model_config()
        return config['models'].get(model_name, {})

    def get_feature_groups(self) -> Dict[str, list]:
        """
        Get feature groups

        Returns:
            Dictionary with feature groups
        """
        config = self.load_features_config()
        return config.get('feature_groups', {})

    def get_all_features(self) -> list:
        """
        Get all features from all groups

        Returns:
            List of all feature names
        """
        feature_groups = self.get_feature_groups()
        all_features = []

        for group, features in feature_groups.items():
            all_features.extend(features)

        return list(set(all_features))  # Remove duplicates

    def get_high_importance_features(self) -> list:
        """
        Get list of high importance features

        Returns:
            List of high importance feature names
        """
        config = self.load_features_config()
        return config.get('high_importance_features', [])

    def get_data_paths(self) -> Dict[str, str]:
        """
        Get data paths from configuration

        Returns:
            Dictionary with data paths
        """
        config = self.load_model_config()
        return config.get('data', {})

    def get_output_paths(self) -> Dict[str, str]:
        """
        Get output paths from configuration

        Returns:
            Dictionary with output paths
        """
        config = self.load_model_config()
        return config.get('output', {})


def get_project_root() -> Path:
    """
    Get project root directory

    Returns:
        Path to project root
    """
    # Assume this file is in src/utils/
    return Path(__file__).parent.parent.parent


def ensure_directories_exist(paths: list):
    """
    Ensure all directories in the list exist

    Args:
        paths: List of directory paths
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
