"""
Data preprocessing module for ML models
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Tuple, Optional, List
import logging
import pickle

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess data for machine learning models"""

    def __init__(self, lookback: int = 60, scaler_type: str = 'minmax'):
        """
        Initialize data preprocessor

        Args:
            lookback: Number of time steps to look back
            scaler_type: Type of scaler ('minmax', 'standard', 'robust')
        """
        self.lookback = lookback
        self.scaler_type = scaler_type

        # Initialize scaler
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}, using MinMaxScaler")
            self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.is_fitted = False
        logger.info(f"DataPreprocessor initialized with lookback={lookback}, scaler={scaler_type}")

    def scale_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale data using the configured scaler

        Args:
            data: Input data (2D array)
            fit: Whether to fit the scaler

        Returns:
            Scaled data
        """
        try:
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            if fit:
                scaled = self.scaler.fit_transform(data)
                self.is_fitted = True
                logger.info(f"Fitted and transformed data with shape {data.shape}")
            else:
                if not self.is_fitted:
                    logger.warning("Scaler not fitted, fitting now...")
                    scaled = self.scaler.fit_transform(data)
                    self.is_fitted = True
                else:
                    scaled = self.scaler.transform(data)
                logger.info(f"Transformed data with shape {data.shape}")

            return scaled

        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            raise

    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data

        Args:
            data: Scaled data

        Returns:
            Original scale data
        """
        try:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Cannot inverse transform.")

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            return self.scaler.inverse_transform(data)

        except Exception as e:
            logger.error(f"Error inverse scaling data: {str(e)}")
            raise

    def create_sequences(self, data: np.ndarray, target_column: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction

        Args:
            data: Input data (scaled)
            target_column: Index of column to predict

        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        try:
            X, y = [], []

            for i in range(self.lookback, len(data)):
                X.append(data[i-self.lookback:i])
                y.append(data[i, target_column])

            X = np.array(X)
            y = np.array(y)

            logger.info(f"Created sequences: X shape={X.shape}, y shape={y.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise

    def prepare_univariate_data(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare univariate time series data (single feature - price)

        Args:
            prices: 1D array of prices

        Returns:
            Tuple of (X, y) ready for LSTM
        """
        try:
            # Scale data
            scaled_data = self.scale_data(prices.reshape(-1, 1), fit=True)

            # Create sequences
            X, y = self.create_sequences(scaled_data, target_column=0)

            # Reshape X for LSTM [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)

            logger.info(f"Prepared univariate data: X shape={X.shape}, y shape={y.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing univariate data: {str(e)}")
            raise

    def prepare_multivariate_data(self, df: pd.DataFrame,
                                  target_col: str = 'close',
                                  feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare multivariate time series data

        Args:
            df: DataFrame with features
            target_col: Name of target column
            feature_cols: List of feature column names (None = all except target)

        Returns:
            Tuple of (X, y) ready for LSTM
        """
        try:
            # Select features
            if feature_cols is None:
                # Use all numeric columns
                feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                # Ensure target column is included
                if target_col not in feature_cols:
                    feature_cols = [target_col] + feature_cols

            # Get data
            data = df[feature_cols].values

            # Scale data
            scaled_data = self.scale_data(data, fit=True)

            # Get target column index
            target_idx = feature_cols.index(target_col)

            # Create sequences
            X, y = self.create_sequences(scaled_data, target_column=target_idx)

            logger.info(f"Prepared multivariate data: X shape={X.shape}, y shape={y.shape}")
            logger.info(f"Features used: {feature_cols}")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing multivariate data: {str(e)}")
            raise

    def train_test_split(self, X: np.ndarray, y: np.ndarray,
                        train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets (time series aware)

        Args:
            X: Feature sequences
            y: Target values
            train_ratio: Ratio of data to use for training

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            split_idx = int(len(X) * train_ratio)

            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]

            logger.info(f"Train/test split: {len(X_train)}/{len(X_test)} samples")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def save_scaler(self, filepath: str):
        """
        Save the fitted scaler to disk

        Args:
            filepath: Path to save scaler
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving scaler: {str(e)}")

    def load_scaler(self, filepath: str):
        """
        Load a fitted scaler from disk

        Args:
            filepath: Path to load scaler from
        """
        try:
            with open(filepath, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
            logger.info(f"Scaler loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")


class SequenceGenerator:
    """Generate sequences for online prediction"""

    def __init__(self, lookback: int):
        """
        Initialize sequence generator

        Args:
            lookback: Number of time steps
        """
        self.lookback = lookback
        self.buffer = []

    def add_data(self, data: np.ndarray):
        """
        Add new data point to buffer

        Args:
            data: New data point (can be single value or feature vector)
        """
        self.buffer.append(data)

        # Keep only lookback + 1 points
        if len(self.buffer) > self.lookback:
            self.buffer.pop(0)

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get current sequence for prediction

        Returns:
            Sequence array if enough data, None otherwise
        """
        if len(self.buffer) < self.lookback:
            return None

        return np.array(self.buffer[-self.lookback:])

    def is_ready(self) -> bool:
        """
        Check if buffer has enough data for prediction

        Returns:
            True if ready, False otherwise
        """
        return len(self.buffer) >= self.lookback


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG
    from ml_trading_bot.data.ingestion import StockDataFetcher
    from ml_trading_bot.features import FeatureEngineer

    logging.config.dictConfig(LOGGING_CONFIG)

    # Fetch and prepare data
    fetcher = StockDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', '2022-01-01', '2024-01-01')

    # Add features
    engineer = FeatureEngineer(df)
    features_df = engineer.add_all_indicators().get_features()

    # Prepare data
    preprocessor = DataPreprocessor(lookback=60)

    # Univariate example
    print("\n=== Univariate Data ===")
    X_uni, y_uni = preprocessor.prepare_univariate_data(features_df['close'].values)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_uni, y_uni)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")

    # Multivariate example
    print("\n=== Multivariate Data ===")
    feature_cols = ['close', 'volume', 'RSI', 'MACD', 'SMA_20', 'BB_upper', 'BB_lower']
    X_multi, y_multi = preprocessor.prepare_multivariate_data(features_df, target_col='close', feature_cols=feature_cols)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_multi, y_multi)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
