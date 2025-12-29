"""
Sliding window retraining system for continuous model updates
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Callable, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SlidingWindowTrainer:
    """Sliding window trainer for continuous model updates"""

    def __init__(self, model_factory: Callable, window_size: int = 60,
                 retrain_frequency: int = 5, min_train_size: int = 100):
        """
        Initialize sliding window trainer

        Args:
            model_factory: Function that creates a new model instance
            window_size: Size of sliding window in days
            retrain_frequency: Retrain every N days
            min_train_size: Minimum samples required for training
        """
        self.model_factory = model_factory
        self.window_size = window_size
        self.retrain_frequency = retrain_frequency
        self.min_train_size = min_train_size

        self.model = None
        self.days_since_retrain = 0
        self.last_retrain_date = None
        self.training_history = []

        logger.info(
            f"SlidingWindowTrainer initialized: window_size={window_size}, "
            f"retrain_frequency={retrain_frequency}"
        )

    def train_on_window(self, data: pd.DataFrame, preprocessor,
                       validation_split: float = 0.2) -> dict:
        """
        Train model on sliding window of data

        Args:
            data: DataFrame with historical data
            preprocessor: DataPreprocessor instance
            validation_split: Validation set ratio

        Returns:
            Dictionary with training metrics
        """
        try:
            # Use only last window_size days
            window_data = data.tail(self.window_size)

            if len(window_data) < self.min_train_size:
                logger.warning(
                    f"Insufficient data for training: {len(window_data)} < {self.min_train_size}"
                )
                return {'error': 'insufficient_data'}

            logger.info(f"Training on window of {len(window_data)} samples")

            # Prepare data
            X, y = preprocessor.prepare_multivariate_data(window_data)

            if len(X) < self.min_train_size:
                logger.warning(f"Insufficient sequences: {len(X)} < {self.min_train_size}")
                return {'error': 'insufficient_sequences'}

            # Split for validation
            split = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            # Create and train model
            self.model = self.model_factory()

            training_metrics = self.model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=50,
                batch_size=32
            )

            # Update state
            self.days_since_retrain = 0
            self.last_retrain_date = datetime.now()

            # Record history
            history_entry = {
                'timestamp': self.last_retrain_date,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'metrics': training_metrics
            }
            self.training_history.append(history_entry)

            logger.info(
                f"Training completed: {len(X_train)} train samples, "
                f"{len(X_val)} val samples"
            )

            return history_entry

        except Exception as e:
            logger.error(f"Error training on window: {str(e)}")
            return {'error': str(e)}

    def update_and_check_retrain(self, data: pd.DataFrame,
                                 preprocessor) -> Tuple[bool, Optional[dict]]:
        """
        Update day counter and check if retraining is needed

        Args:
            data: Current data
            preprocessor: DataPreprocessor instance

        Returns:
            Tuple of (retrained, metrics)
        """
        try:
            self.days_since_retrain += 1

            if self.days_since_retrain >= self.retrain_frequency:
                logger.info(
                    f"Retraining triggered: {self.days_since_retrain} days "
                    f"since last retrain"
                )

                metrics = self.train_on_window(data, preprocessor)

                if 'error' not in metrics:
                    return True, metrics
                else:
                    logger.warning(f"Retraining failed: {metrics['error']}")
                    return False, metrics

            return False, None

        except Exception as e:
            logger.error(f"Error in update_and_check_retrain: {str(e)}")
            return False, None

    def get_model(self):
        """
        Get current model

        Returns:
            Current model instance
        """
        if self.model is None:
            logger.warning("Model not trained yet")
        return self.model

    def get_training_history(self) -> pd.DataFrame:
        """
        Get training history as DataFrame

        Returns:
            DataFrame with training history
        """
        if not self.training_history:
            return pd.DataFrame()

        return pd.DataFrame(self.training_history)

    def save_state(self, filepath: str):
        """
        Save trainer state

        Args:
            filepath: Path to save state
        """
        try:
            import pickle

            state = {
                'window_size': self.window_size,
                'retrain_frequency': self.retrain_frequency,
                'days_since_retrain': self.days_since_retrain,
                'last_retrain_date': self.last_retrain_date,
                'training_history': self.training_history
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Trainer state saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")

    def load_state(self, filepath: str):
        """
        Load trainer state

        Args:
            filepath: Path to load state from
        """
        try:
            import pickle

            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.window_size = state['window_size']
            self.retrain_frequency = state['retrain_frequency']
            self.days_since_retrain = state['days_since_retrain']
            self.last_retrain_date = state['last_retrain_date']
            self.training_history = state['training_history']

            logger.info(f"Trainer state loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")


class AdaptiveWindowTrainer(SlidingWindowTrainer):
    """Adaptive sliding window trainer that adjusts window size based on performance"""

    def __init__(self, model_factory: Callable, initial_window_size: int = 60,
                 retrain_frequency: int = 5, min_window_size: int = 30,
                 max_window_size: int = 180):
        """
        Initialize adaptive window trainer

        Args:
            model_factory: Function that creates a new model instance
            initial_window_size: Initial window size
            retrain_frequency: Retrain frequency
            min_window_size: Minimum window size
            max_window_size: Maximum window size
        """
        super().__init__(model_factory, initial_window_size, retrain_frequency)

        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.performance_history = []

        logger.info(
            f"AdaptiveWindowTrainer initialized: "
            f"window_size={initial_window_size} (min={min_window_size}, max={max_window_size})"
        )

    def adjust_window_size(self, recent_performance: float):
        """
        Adjust window size based on recent performance

        Args:
            recent_performance: Recent model performance metric (e.g., SMAPE)
        """
        try:
            self.performance_history.append(recent_performance)

            # Only adjust after enough history
            if len(self.performance_history) < 3:
                return

            # Check if performance is improving or degrading
            recent_avg = np.mean(self.performance_history[-3:])

            if len(self.performance_history) >= 6:
                older_avg = np.mean(self.performance_history[-6:-3])

                # If performance degrading, increase window size (more data)
                if recent_avg > older_avg * 1.1:
                    new_window_size = min(
                        self.window_size + 10,
                        self.max_window_size
                    )

                    if new_window_size != self.window_size:
                        logger.info(
                            f"Performance degrading, increasing window size: "
                            f"{self.window_size} -> {new_window_size}"
                        )
                        self.window_size = new_window_size

                # If performance improving, can try smaller window (more responsive)
                elif recent_avg < older_avg * 0.9:
                    new_window_size = max(
                        self.window_size - 10,
                        self.min_window_size
                    )

                    if new_window_size != self.window_size:
                        logger.info(
                            f"Performance improving, decreasing window size: "
                            f"{self.window_size} -> {new_window_size}"
                        )
                        self.window_size = new_window_size

        except Exception as e:
            logger.error(f"Error adjusting window size: {str(e)}")


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG
    from ml_trading_bot.data.ingestion import StockDataFetcher
    from ml_trading_bot.features import FeatureEngineer
    from ml_trading_bot.data.preprocessing import DataPreprocessor
    from ml_trading_bot.models import LSTMModelBuilder

    logging.config.dictConfig(LOGGING_CONFIG)

    print("=== Sliding Window Training Example ===\n")

    # Fetch data
    print("Fetching data...")
    fetcher = StockDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', '2022-01-01', '2024-01-01')

    # Engineer features
    print("Engineering features...")
    engineer = FeatureEngineer(df)
    features_df = engineer.add_all_indicators().get_features()

    # Create preprocessor
    preprocessor = DataPreprocessor(lookback=60)

    # Define model factory
    def model_factory():
        return LSTMModelBuilder(
            lookback=60,
            n_features=7,
            units=100,
            dropout_rate=0.2
        )

    # Create sliding window trainer
    print("\nCreating sliding window trainer...")
    trainer = SlidingWindowTrainer(
        model_factory=model_factory,
        window_size=60,
        retrain_frequency=5
    )

    # Train on window
    print("\nTraining on sliding window...")
    metrics = trainer.train_on_window(features_df, preprocessor)

    if 'error' not in metrics:
        print(f"Training completed successfully!")
        print(f"Train samples: {metrics['train_samples']}")
        print(f"Val samples: {metrics['val_samples']}")
    else:
        print(f"Training failed: {metrics['error']}")

    # Simulate daily updates
    print("\nSimulating daily updates...")
    for day in range(10):
        retrained, metrics = trainer.update_and_check_retrain(features_df, preprocessor)

        if retrained:
            print(f"Day {day+1}: Model retrained!")
        else:
            print(f"Day {day+1}: No retraining needed (days since retrain: {trainer.days_since_retrain})")

    # Show training history
    print("\nTraining History:")
    print(trainer.get_training_history())
