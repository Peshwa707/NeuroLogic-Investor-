"""
Online learning models using River library
"""
from river import linear_model, preprocessing, metrics, ensemble, tree
import logging
import pickle
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


class OnlineLearner:
    """Online learning classifier for real-time adaptation"""

    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize online learner

        Args:
            model_type: Type of model ('logistic', 'pa', 'adaptive_forest')
        """
        self.model_type = model_type

        # Initialize model with preprocessing pipeline
        if model_type == 'logistic':
            self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        elif model_type == 'pa':  # Passive-Aggressive
            self.model = preprocessing.StandardScaler() | linear_model.PAClassifier()
        elif model_type == 'adaptive_forest':
            self.model = ensemble.AdaptiveRandomForestClassifier(n_models=10)
        else:
            logger.warning(f"Unknown model type: {model_type}, using logistic regression")
            self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()

        # Metrics
        self.accuracy_metric = metrics.Accuracy()
        self.auc_metric = metrics.ROCAUC()
        self.precision_metric = metrics.Precision()
        self.recall_metric = metrics.Recall()

        logger.info(f"OnlineLearner initialized with model type: {model_type}")

    def learn_one(self, features: dict, target: int):
        """
        Update model with single data point (online learning)

        Args:
            features: Dictionary of feature name -> value
            target: Target label (0 or 1)
        """
        try:
            # Get prediction before learning
            y_pred = self.model.predict_proba_one(features)

            # Update metrics
            if y_pred:
                prob_positive = y_pred.get(True, y_pred.get(1, 0.5))
                self.accuracy_metric.update(target, self.model.predict_one(features))
                self.auc_metric.update(target, prob_positive)
                self.precision_metric.update(target, self.model.predict_one(features))
                self.recall_metric.update(target, self.model.predict_one(features))

            # Learn from new data
            self.model.learn_one(features, target)

        except Exception as e:
            logger.error(f"Error in learn_one: {str(e)}")

    def predict_one(self, features: dict) -> int:
        """
        Predict class for single data point

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Predicted class (0 or 1)
        """
        try:
            prediction = self.model.predict_one(features)
            return prediction if prediction is not None else 0

        except Exception as e:
            logger.error(f"Error in predict_one: {str(e)}")
            return 0

    def predict_proba_one(self, features: dict) -> float:
        """
        Predict probability for single data point

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Probability of positive class
        """
        try:
            proba = self.model.predict_proba_one(features)

            if proba is None:
                return 0.5

            # Extract probability (handle different key formats)
            if True in proba:
                return proba[True]
            elif 1 in proba:
                return proba[1]
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Error in predict_proba_one: {str(e)}")
            return 0.5

    def get_performance(self) -> Dict[str, float]:
        """
        Get current performance metrics

        Returns:
            Dictionary of metrics
        """
        return {
            'accuracy': self.accuracy_metric.get(),
            'roc_auc': self.auc_metric.get(),
            'precision': self.precision_metric.get(),
            'recall': self.recall_metric.get()
        }

    def predict_signal(self, features: dict, buy_threshold: float = 0.6,
                      sell_threshold: float = 0.4) -> str:
        """
        Predict trading signal

        Args:
            features: Dictionary of feature name -> value
            buy_threshold: Threshold for BUY signal
            sell_threshold: Threshold for SELL signal

        Returns:
            Trading signal ('BUY', 'SELL', 'HOLD')
        """
        try:
            proba = self.predict_proba_one(features)

            if proba > buy_threshold:
                return 'BUY'
            elif proba < sell_threshold:
                return 'SELL'
            else:
                return 'HOLD'

        except Exception as e:
            logger.error(f"Error predicting signal: {str(e)}")
            return 'HOLD'

    def save_model(self, filepath: str):
        """
        Save model to disk

        Args:
            filepath: Path to save model
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'model_type': self.model_type,
                    'metrics': self.get_performance()
                }, f)

            logger.info(f"Online model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, filepath: str):
        """
        Load model from disk

        Args:
            filepath: Path to load model from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.model = data['model']
            self.model_type = data['model_type']

            logger.info(f"Online model loaded from {filepath}")
            logger.info(f"Previous metrics: {data.get('metrics', {})}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")


class OnlineRegressor:
    """Online learning regressor for price prediction"""

    def __init__(self):
        """Initialize online regressor"""
        self.model = preprocessing.StandardScaler() | linear_model.LinearRegression()

        # Metrics
        self.mae_metric = metrics.MAE()
        self.rmse_metric = metrics.RMSE()

        logger.info("OnlineRegressor initialized")

    def learn_one(self, features: dict, target: float):
        """
        Update model with single data point

        Args:
            features: Dictionary of feature name -> value
            target: Target value (price)
        """
        try:
            # Get prediction before learning
            y_pred = self.model.predict_one(features)

            # Update metrics
            if y_pred is not None:
                self.mae_metric.update(target, y_pred)
                self.rmse_metric.update(target, y_pred)

            # Learn from new data
            self.model.learn_one(features, target)

        except Exception as e:
            logger.error(f"Error in learn_one: {str(e)}")

    def predict_one(self, features: dict) -> float:
        """
        Predict value for single data point

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Predicted value
        """
        try:
            prediction = self.model.predict_one(features)
            return prediction if prediction is not None else 0.0

        except Exception as e:
            logger.error(f"Error in predict_one: {str(e)}")
            return 0.0

    def get_performance(self) -> Dict[str, float]:
        """
        Get current performance metrics

        Returns:
            Dictionary of metrics
        """
        return {
            'mae': self.mae_metric.get(),
            'rmse': self.rmse_metric.get()
        }


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG
    import pandas as pd
    import numpy as np

    logging.config.dictConfig(LOGGING_CONFIG)

    print("=== Online Learning Example ===\n")

    # Create online learner
    learner = OnlineLearner(model_type='logistic')

    # Simulate streaming data
    np.random.seed(42)
    n_samples = 1000

    print("Simulating online learning with streaming data...\n")

    for i in range(n_samples):
        # Generate synthetic features
        features = {
            'rsi': np.random.uniform(0, 100),
            'macd': np.random.uniform(-2, 2),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'price_change_pct': np.random.uniform(-5, 5)
        }

        # Generate target (1 if price will increase, 0 otherwise)
        # Simple rule: if RSI < 30 and MACD > 0, predict increase
        target = int(features['rsi'] < 40 and features['macd'] > 0)

        # Learn from this data point
        learner.learn_one(features, target)

        # Print performance every 100 samples
        if (i + 1) % 200 == 0:
            perf = learner.get_performance()
            print(f"Sample {i+1}: Accuracy={perf['accuracy']:.4f}, ROC-AUC={perf['roc_auc']:.4f}")

    # Final performance
    print(f"\nFinal Performance:")
    final_perf = learner.get_performance()
    for metric, value in final_perf.items():
        print(f"  {metric}: {value:.4f}")

    # Test prediction
    print(f"\nTest Predictions:")
    test_features = [
        {'rsi': 25, 'macd': 0.5, 'volume_ratio': 1.2, 'price_change_pct': 1.0},
        {'rsi': 75, 'macd': -0.5, 'volume_ratio': 0.8, 'price_change_pct': -2.0}
    ]

    for features in test_features:
        signal = learner.predict_signal(features)
        proba = learner.predict_proba_one(features)
        print(f"  Features: {features}")
        print(f"  Signal: {signal}, Probability: {proba:.4f}\n")
