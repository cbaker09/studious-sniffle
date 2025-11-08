"""
Logging utilities for loan default prediction system
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "loan_default_prediction",
    log_level: str = "INFO",
    log_file: str = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_model_training_start(logger: logging.Logger, model_name: str):
    """Log model training start"""
    logger.info("=" * 60)
    logger.info(f"Starting training for {model_name}")
    logger.info("=" * 60)


def log_model_training_complete(
    logger: logging.Logger,
    model_name: str,
    metrics: dict
):
    """
    Log model training completion with metrics

    Args:
        logger: Logger instance
        model_name: Name of the model
        metrics: Dictionary of metrics
    """
    logger.info(f"Training complete for {model_name}")
    logger.info("Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")


def log_data_loading(logger: logging.Logger, n_rows: int, n_features: int):
    """
    Log data loading information

    Args:
        logger: Logger instance
        n_rows: Number of rows loaded
        n_features: Number of features
    """
    logger.info(f"Loaded {n_rows:,} rows with {n_features} features")


def log_feature_engineering(logger: logging.Logger, message: str):
    """
    Log feature engineering step

    Args:
        logger: Logger instance
        message: Feature engineering message
    """
    logger.info(f"Feature Engineering: {message}")


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """
    Log an error with context

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context about the error
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg, exc_info=True)


class TimedOperation:
    """Context manager for timing operations"""

    def __init__(self, logger: logging.Logger, operation_name: str):
        """
        Initialize timed operation

        Args:
            logger: Logger instance
            operation_name: Name of the operation being timed
        """
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        """Start timing"""
        self.logger.info(f"Starting: {self.operation_name}")
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration"""
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(
                f"Completed: {self.operation_name} "
                f"(Duration: {duration:.2f}s)"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation_name} "
                f"(Duration: {duration:.2f}s)"
            )
        return False  # Don't suppress exceptions
