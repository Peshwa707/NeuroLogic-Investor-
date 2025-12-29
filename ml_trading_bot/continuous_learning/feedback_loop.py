"""
Continuous learning feedback loop and model drift detection
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Callable
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ContinuousLearningManager:
    """Manage continuous learning with feedback loop"""

    def __init__(self, model, error_threshold: float = 0.05,
                 window_size: int = 50):
        """
        Initialize continuous learning manager

        Args:
            model: Model instance (must have fit/predict methods)
            error_threshold: Error threshold for triggering retraining
            window_size: Window size for performance tracking
        """
        self.model = model
        self.error_threshold = error_threshold
        self.window_size = window_size

        # Performance tracking
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # Metrics history
        self.metrics_history = []

        logger.info(
            f"ContinuousLearningManager initialized with "
            f"error_threshold={error_threshold}, window_size={window_size}"
        )

    def add_prediction(self, prediction: float, actual: float,
                      timestamp: Optional[datetime] = None):
        """
        Record prediction and actual value

        Args:
            prediction: Predicted value
            actual: Actual value
            timestamp: Timestamp (defaults to now)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()

            # Calculate error
            error = abs(prediction - actual) / actual if actual != 0 else 0

            # Store values
            self.predictions.append(prediction)
            self.actuals.append(actual)
            self.errors.append(error)
            self.timestamps.append(timestamp)

            logger.debug(
                f"Added prediction: pred={prediction:.4f}, "
                f"actual={actual:.4f}, error={error:.4f}"
            )

        except Exception as e:
            logger.error(f"Error adding prediction: {str(e)}")

    def calculate_smape(self) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error

        Returns:
            SMAPE value (0-100)
        """
        if not self.predictions:
            return 0.0

        try:
            predictions = np.array(list(self.predictions))
            actuals = np.array(list(self.actuals))

            numerator = np.abs(predictions - actuals)
            denominator = (np.abs(predictions) + np.abs(actuals)) / 2

            # Avoid division by zero
            denominator = np.where(denominator == 0, 1e-10, denominator)

            smape = np.mean(numerator / denominator) * 100

            return float(smape)

        except Exception as e:
            logger.error(f"Error calculating SMAPE: {str(e)}")
            return 0.0

    def calculate_mape(self) -> float:
        """
        Calculate Mean Absolute Percentage Error

        Returns:
            MAPE value (0-100)
        """
        if not self.predictions:
            return 0.0

        try:
            predictions = np.array(list(self.predictions))
            actuals = np.array(list(self.actuals))

            # Avoid division by zero
            actuals = np.where(actuals == 0, 1e-10, actuals)

            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

            return float(mape)

        except Exception as e:
            logger.error(f"Error calculating MAPE: {str(e)}")
            return 0.0

    def calculate_directional_accuracy(self) -> float:
        """
        Calculate percentage of correct direction predictions

        Returns:
            Directional accuracy (0-1)
        """
        if len(self.predictions) < 2:
            return 0.0

        try:
            predictions = np.array(list(self.predictions))
            actuals = np.array(list(self.actuals))

            # Calculate direction changes
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0

            # Calculate accuracy
            accuracy = np.mean(pred_direction == actual_direction)

            return float(accuracy)

        except Exception as e:
            logger.error(f"Error calculating directional accuracy: {str(e)}")
            return 0.0

    def get_current_metrics(self) -> dict:
        """
        Get current performance metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'smape': self.calculate_smape(),
            'mape': self.calculate_mape(),
            'directional_accuracy': self.calculate_directional_accuracy(),
            'mean_error': np.mean(list(self.errors)) if self.errors else 0.0,
            'std_error': np.std(list(self.errors)) if self.errors else 0.0,
            'n_samples': len(self.predictions),
            'timestamp': datetime.now()
        }

        return metrics

    def should_retrain(self) -> bool:
        """
        Check if model performance has degraded and retraining is needed

        Returns:
            True if retraining is recommended, False otherwise
        """
        if len(self.errors) < self.window_size:
            return False

        try:
            # Check recent error
            recent_error = np.mean(list(self.errors)[-self.window_size:])

            if recent_error > self.error_threshold:
                logger.warning(
                    f"Model performance degraded: recent error={recent_error:.4f} "
                    f"exceeds threshold={self.error_threshold:.4f}"
                )
                return True

            # Check error trend (increasing errors over time)
            if len(self.errors) >= self.window_size:
                first_half = list(self.errors)[:self.window_size//2]
                second_half = list(self.errors)[self.window_size//2:]

                if np.mean(second_half) > np.mean(first_half) * 1.5:
                    logger.warning("Model error is increasing over time")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking retrain condition: {str(e)}")
            return False

    def retrain(self, X_new: np.ndarray, y_new: np.ndarray,
               retrain_callback: Optional[Callable] = None) -> bool:
        """
        Retrain model on new data

        Args:
            X_new: New training features
            y_new: New training targets
            retrain_callback: Optional callback function for custom retraining

        Returns:
            True if retraining successful, False otherwise
        """
        try:
            logger.info(f"Retraining model on {len(X_new)} new samples...")

            if retrain_callback:
                retrain_callback(self.model, X_new, y_new)
            else:
                # Default retraining (works for scikit-learn style models)
                self.model.fit(X_new, y_new)

            # Clear error tracking after retraining
            self.errors.clear()
            self.predictions.clear()
            self.actuals.clear()
            self.timestamps.clear()

            logger.info("Retraining completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
            return False

    def save_metrics_history(self):
        """Save current metrics to history"""
        metrics = self.get_current_metrics()
        self.metrics_history.append(metrics)

        logger.info(f"Saved metrics: SMAPE={metrics['smape']:.2f}%, "
                   f"Directional Acc={metrics['directional_accuracy']:.2%}")

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get metrics history as DataFrame

        Returns:
            DataFrame with metrics history
        """
        if not self.metrics_history:
            return pd.DataFrame()

        return pd.DataFrame(self.metrics_history)

    def reset(self):
        """Reset all tracked data"""
        self.predictions.clear()
        self.actuals.clear()
        self.errors.clear()
        self.timestamps.clear()
        self.metrics_history.clear()

        logger.info("ContinuousLearningManager reset")


class ModelDriftDetector:
    """Detect model drift and performance degradation"""

    def __init__(self, baseline_metrics: dict, alert_threshold: float = 0.2):
        """
        Initialize drift detector

        Args:
            baseline_metrics: Baseline performance metrics
            alert_threshold: Threshold for drift alert (20% degradation by default)
        """
        self.baseline_metrics = baseline_metrics
        self.alert_threshold = alert_threshold

        logger.info(f"ModelDriftDetector initialized with baseline: {baseline_metrics}")

    def detect_drift(self, current_metrics: dict) -> dict:
        """
        Detect model drift by comparing current metrics to baseline

        Args:
            current_metrics: Current performance metrics

        Returns:
            Dictionary with drift detection results
        """
        try:
            drift_detected = False
            degraded_metrics = []

            for metric_name, baseline_value in self.baseline_metrics.items():
                if metric_name not in current_metrics:
                    continue

                current_value = current_metrics[metric_name]

                # Skip non-numeric values
                if not isinstance(current_value, (int, float)):
                    continue

                # Calculate relative change
                if baseline_value != 0:
                    rel_change = (current_value - baseline_value) / baseline_value
                else:
                    rel_change = 0

                # For metrics where lower is better (SMAPE, MAPE, error)
                if metric_name in ['smape', 'mape', 'mean_error', 'std_error']:
                    if rel_change > self.alert_threshold:
                        drift_detected = True
                        degraded_metrics.append({
                            'metric': metric_name,
                            'baseline': baseline_value,
                            'current': current_value,
                            'change': rel_change
                        })

                # For metrics where higher is better (accuracy)
                elif metric_name in ['directional_accuracy', 'accuracy', 'roc_auc']:
                    if rel_change < -self.alert_threshold:
                        drift_detected = True
                        degraded_metrics.append({
                            'metric': metric_name,
                            'baseline': baseline_value,
                            'current': current_value,
                            'change': rel_change
                        })

            result = {
                'drift_detected': drift_detected,
                'degraded_metrics': degraded_metrics,
                'timestamp': datetime.now()
            }

            if drift_detected:
                logger.warning(f"Model drift detected! Degraded metrics: {degraded_metrics}")
            else:
                logger.info("No model drift detected")

            return result

        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            return {'drift_detected': False, 'degraded_metrics': [], 'timestamp': datetime.now()}


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG

    logging.config.dictConfig(LOGGING_CONFIG)

    print("=== Continuous Learning Example ===\n")

    # Mock model
    class MockModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return X  # Simple identity prediction for testing

    model = MockModel()

    # Create continuous learning manager
    manager = ContinuousLearningManager(model, error_threshold=0.05, window_size=50)

    # Simulate predictions and actuals
    print("Simulating predictions...\n")
    np.random.seed(42)

    for i in range(100):
        # Simulate prediction with increasing error over time
        actual = 100 + np.random.randn() * 5
        prediction = actual + np.random.randn() * 2 + (i * 0.1)  # Error increases over time

        manager.add_prediction(prediction, actual)

        # Check metrics every 25 samples
        if (i + 1) % 25 == 0:
            metrics = manager.get_current_metrics()
            print(f"Sample {i+1}:")
            print(f"  SMAPE: {metrics['smape']:.2f}%")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
            print(f"  Should retrain: {manager.should_retrain()}\n")

            manager.save_metrics_history()

    # Get metrics DataFrame
    print("\nMetrics History:")
    print(manager.get_metrics_dataframe())

    # Drift detection example
    print("\n=== Drift Detection Example ===\n")

    baseline_metrics = {
        'smape': 2.0,
        'directional_accuracy': 0.75
    }

    drift_detector = ModelDriftDetector(baseline_metrics, alert_threshold=0.2)

    current_metrics = manager.get_current_metrics()
    drift_result = drift_detector.detect_drift(current_metrics)

    print(f"Drift detected: {drift_result['drift_detected']}")
    if drift_result['degraded_metrics']:
        print("Degraded metrics:")
        for metric in drift_result['degraded_metrics']:
            print(f"  {metric['metric']}: {metric['baseline']:.4f} -> {metric['current']:.4f} "
                  f"({metric['change']:.2%} change)")
