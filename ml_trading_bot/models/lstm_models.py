"""
LSTM models for time series prediction
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import logging
from typing import Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LSTMModelBuilder:
    """Build and train LSTM models for stock/crypto prediction"""

    def __init__(self, lookback: int, n_features: int = 1,
                 units: int = 100, dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM model builder

        Args:
            lookback: Number of time steps to look back
            n_features: Number of features
            units: Number of LSTM units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.lookback = lookback
        self.n_features = n_features
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None

        logger.info(
            f"LSTMModelBuilder initialized: "
            f"lookback={lookback}, features={n_features}, units={units}"
        )

    def create_univariate_lstm(self) -> Model:
        """
        Create univariate LSTM model (single feature - price)

        Returns:
            Keras model
        """
        try:
            model = Sequential([
                LSTM(self.units, return_sequences=True,
                     input_shape=(self.lookback, 1)),
                Dropout(self.dropout_rate),
                LSTM(self.units, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(25, activation='relu'),
                Dense(1)
            ])

            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )

            self.model = model
            logger.info("Created univariate LSTM model")
            return model

        except Exception as e:
            logger.error(f"Error creating univariate LSTM: {str(e)}")
            raise

    def create_multivariate_lstm(self) -> Model:
        """
        Create multivariate LSTM model (multiple features)

        Returns:
            Keras model
        """
        try:
            model = Sequential([
                LSTM(self.units, return_sequences=True,
                     input_shape=(self.lookback, self.n_features)),
                Dropout(self.dropout_rate),
                LSTM(self.units, return_sequences=True),
                Dropout(self.dropout_rate),
                LSTM(self.units // 2, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(50, activation='relu'),
                Dense(1)
            ])

            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='huber',  # More robust to outliers
                metrics=['mae', 'mse']
            )

            self.model = model
            logger.info(f"Created multivariate LSTM model with {self.n_features} features")
            return model

        except Exception as e:
            logger.error(f"Error creating multivariate LSTM: {str(e)}")
            raise

    def create_bidirectional_lstm(self) -> Model:
        """
        Create bidirectional LSTM model

        Returns:
            Keras model
        """
        try:
            model = Sequential([
                Bidirectional(LSTM(self.units, return_sequences=True),
                            input_shape=(self.lookback, self.n_features)),
                Dropout(self.dropout_rate),
                Bidirectional(LSTM(self.units // 2, return_sequences=False)),
                Dropout(self.dropout_rate),
                Dense(50, activation='relu'),
                Dense(1)
            ])

            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='huber',
                metrics=['mae', 'mse']
            )

            self.model = model
            logger.info("Created bidirectional LSTM model")
            return model

        except Exception as e:
            logger.error(f"Error creating bidirectional LSTM: {str(e)}")
            raise

    def create_stateful_lstm(self, batch_size: int) -> Model:
        """
        Create stateful LSTM model for continuous learning

        Args:
            batch_size: Batch size (must be fixed for stateful LSTM)

        Returns:
            Keras model
        """
        try:
            model = Sequential([
                LSTM(self.units, stateful=True, return_sequences=True,
                     batch_input_shape=(batch_size, self.lookback, self.n_features)),
                Dropout(self.dropout_rate),
                LSTM(self.units // 2, stateful=True, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(1)
            ])

            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )

            self.model = model
            logger.info(f"Created stateful LSTM model with batch_size={batch_size}")
            return model

        except Exception as e:
            logger.error(f"Error creating stateful LSTM: {str(e)}")
            raise

    def create_attention_lstm(self) -> Model:
        """
        Create LSTM model with attention mechanism

        Returns:
            Keras model
        """
        try:
            # This is a simplified attention model
            inputs = Input(shape=(self.lookback, self.n_features))

            # LSTM layer
            lstm_out = LSTM(self.units, return_sequences=True)(inputs)
            lstm_out = Dropout(self.dropout_rate)(lstm_out)

            # Attention mechanism (simplified)
            attention = Dense(1, activation='tanh')(lstm_out)
            attention = tf.nn.softmax(attention, axis=1)
            attention_out = tf.reduce_sum(lstm_out * attention, axis=1)

            # Dense layers
            dense = Dense(50, activation='relu')(attention_out)
            outputs = Dense(1)(dense)

            model = Model(inputs=inputs, outputs=outputs)

            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='huber',
                metrics=['mae', 'mse']
            )

            self.model = model
            logger.info("Created attention-based LSTM model")
            return model

        except Exception as e:
            logger.error(f"Error creating attention LSTM: {str(e)}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32,
             save_path: Optional[str] = None) -> dict:
        """
        Train the LSTM model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            save_path: Path to save best model

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_*_lstm() first.")

        try:
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]

            if save_path:
                callbacks.append(
                    ModelCheckpoint(
                        save_path,
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    )
                )

            logger.info(f"Training LSTM model for {epochs} epochs...")

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Training completed")
            return history.history

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not created or trained")

        try:
            predictions = self.model.predict(X, verbose=0)
            return predictions.flatten()

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not created or trained")

        try:
            results = self.model.evaluate(X_test, y_test, verbose=0)

            metrics = {
                'loss': results[0],
                'mae': results[1] if len(results) > 1 else None,
                'mse': results[2] if len(results) > 2 else None
            }

            logger.info(f"Evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, filepath: str):
        """
        Save model to disk

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not created")

        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath: str):
        """
        Load model from disk

        Args:
            filepath: Path to model file
        """
        try:
            self.model = keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_model_summary(self) -> str:
        """
        Get model architecture summary

        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not created"

        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG, MODEL_CONFIG
    from ml_trading_bot.data.ingestion import StockDataFetcher
    from ml_trading_bot.features import FeatureEngineer
    from ml_trading_bot.data.preprocessing import DataPreprocessor

    logging.config.dictConfig(LOGGING_CONFIG)

    # Fetch and prepare data
    print("Fetching data...")
    fetcher = StockDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', '2022-01-01', '2024-01-01')

    # Engineer features
    print("Engineering features...")
    engineer = FeatureEngineer(df)
    features_df = engineer.add_all_indicators().get_features()

    # Prepare data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(lookback=60)

    # Use multiple features
    feature_cols = ['close', 'volume', 'RSI', 'MACD', 'SMA_20', 'BB_upper', 'BB_lower']
    X, y = preprocessor.prepare_multivariate_data(features_df, target_col='close', feature_cols=feature_cols)

    # Split data
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, train_ratio=0.8)
    X_train, X_val, y_train, y_val = preprocessor.train_test_split(X_train, y_train, train_ratio=0.8)

    print(f"\nData shapes:")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val: X={X_val.shape}, y={y_val.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")

    # Create and train model
    print("\nCreating model...")
    model_builder = LSTMModelBuilder(
        lookback=60,
        n_features=len(feature_cols),
        units=100,
        dropout_rate=0.2
    )

    model = model_builder.create_multivariate_lstm()
    print(model_builder.get_model_summary())

    print("\nTraining model...")
    history = model_builder.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)

    # Evaluate
    print("\nEvaluating model...")
    metrics = model_builder.evaluate(X_test, y_test)
    print(f"Test metrics: {metrics}")

    # Make predictions
    print("\nMaking predictions...")
    predictions = model_builder.predict(X_test[:5])
    print(f"Predictions: {predictions}")
    print(f"Actuals: {y_test[:5]}")
