"""
Technical indicators feature engineering module
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer technical indicators and features from OHLCV data"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer

        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def add_moving_averages(self, windows: List[int] = [5, 10, 20, 50, 200]) -> 'FeatureEngineer':
        """
        Add Simple and Exponential Moving Averages

        Args:
            windows: List of window sizes

        Returns:
            Self for method chaining
        """
        try:
            for window in windows:
                # Skip if not enough data
                if len(self.df) < window:
                    logger.warning(f"Not enough data for MA window {window}, skipping")
                    continue

                self.df[f'SMA_{window}'] = self.df['close'].rolling(window=window).mean()
                self.df[f'EMA_{window}'] = self.df['close'].ewm(span=window, adjust=False).mean()

                # Add price to MA ratio (useful feature)
                self.df[f'price_to_SMA_{window}'] = self.df['close'] / self.df[f'SMA_{window}']

            logger.info(f"Added moving averages for windows: {windows}")
            return self

        except Exception as e:
            logger.error(f"Error adding moving averages: {str(e)}")
            return self

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> 'FeatureEngineer':
        """
        Add MACD (Moving Average Convergence Divergence)

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Self for method chaining
        """
        try:
            exp1 = self.df['close'].ewm(span=fast, adjust=False).mean()
            exp2 = self.df['close'].ewm(span=slow, adjust=False).mean()

            self.df['MACD'] = exp1 - exp2
            self.df['MACD_signal'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()
            self.df['MACD_histogram'] = self.df['MACD'] - self.df['MACD_signal']

            # Add MACD signal crossover (1 for bullish, -1 for bearish, 0 for no cross)
            self.df['MACD_crossover'] = 0
            macd_diff = self.df['MACD'] - self.df['MACD_signal']
            self.df.loc[macd_diff > 0, 'MACD_crossover'] = 1
            self.df.loc[macd_diff < 0, 'MACD_crossover'] = -1

            logger.info("Added MACD indicators")
            return self

        except Exception as e:
            logger.error(f"Error adding MACD: {str(e)}")
            return self

    def add_bollinger_bands(self, window: int = 20, num_std: int = 2) -> 'FeatureEngineer':
        """
        Add Bollinger Bands

        Args:
            window: Rolling window size
            num_std: Number of standard deviations

        Returns:
            Self for method chaining
        """
        try:
            sma = self.df['close'].rolling(window=window).mean()
            std = self.df['close'].rolling(window=window).std()

            self.df['BB_upper'] = sma + (std * num_std)
            self.df['BB_middle'] = sma
            self.df['BB_lower'] = sma - (std * num_std)

            # Bollinger Band width (volatility indicator)
            self.df['BB_width'] = (self.df['BB_upper'] - self.df['BB_lower']) / self.df['BB_middle']

            # %B indicator (position within bands)
            self.df['BB_pct'] = (self.df['close'] - self.df['BB_lower']) / (self.df['BB_upper'] - self.df['BB_lower'])

            logger.info("Added Bollinger Bands")
            return self

        except Exception as e:
            logger.error(f"Error adding Bollinger Bands: {str(e)}")
            return self

    def add_rsi(self, window: int = 14) -> 'FeatureEngineer':
        """
        Add Relative Strength Index

        Args:
            window: RSI period

        Returns:
            Self for method chaining
        """
        try:
            delta = self.df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            rs = gain / loss
            self.df['RSI'] = 100 - (100 / (1 + rs))

            # Add RSI overbought/oversold signals
            self.df['RSI_signal'] = 0
            self.df.loc[self.df['RSI'] > 70, 'RSI_signal'] = -1  # Overbought
            self.df.loc[self.df['RSI'] < 30, 'RSI_signal'] = 1   # Oversold

            logger.info("Added RSI")
            return self

        except Exception as e:
            logger.error(f"Error adding RSI: {str(e)}")
            return self

    def add_stochastic_oscillator(self, k_window: int = 14, d_window: int = 3) -> 'FeatureEngineer':
        """
        Add Stochastic Oscillator

        Args:
            k_window: %K period
            d_window: %D period (smoothing)

        Returns:
            Self for method chaining
        """
        try:
            # %K
            low_min = self.df['low'].rolling(window=k_window).min()
            high_max = self.df['high'].rolling(window=k_window).max()

            self.df['Stoch_K'] = 100 * (self.df['close'] - low_min) / (high_max - low_min)

            # %D (smoothed %K)
            self.df['Stoch_D'] = self.df['Stoch_K'].rolling(window=d_window).mean()

            logger.info("Added Stochastic Oscillator")
            return self

        except Exception as e:
            logger.error(f"Error adding Stochastic Oscillator: {str(e)}")
            return self

    def add_atr(self, window: int = 14) -> 'FeatureEngineer':
        """
        Add Average True Range (volatility indicator)

        Args:
            window: ATR period

        Returns:
            Self for method chaining
        """
        try:
            high_low = self.df['high'] - self.df['low']
            high_close = np.abs(self.df['high'] - self.df['close'].shift())
            low_close = np.abs(self.df['low'] - self.df['close'].shift())

            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)

            self.df['ATR'] = true_range.rolling(window=window).mean()

            # ATR as percentage of price (normalized volatility)
            self.df['ATR_pct'] = (self.df['ATR'] / self.df['close']) * 100

            logger.info("Added ATR")
            return self

        except Exception as e:
            logger.error(f"Error adding ATR: {str(e)}")
            return self

    def add_adx(self, window: int = 14) -> 'FeatureEngineer':
        """
        Add Average Directional Index (trend strength)

        Args:
            window: ADX period

        Returns:
            Self for method chaining
        """
        try:
            # Use pandas_ta for ADX
            adx_df = ta.adx(
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                length=window
            )

            if adx_df is not None and not adx_df.empty:
                self.df[f'ADX_{window}'] = adx_df[f'ADX_{window}']
                self.df[f'DMP_{window}'] = adx_df[f'DMP_{window}']
                self.df[f'DMN_{window}'] = adx_df[f'DMN_{window}']

                logger.info("Added ADX")

            return self

        except Exception as e:
            logger.error(f"Error adding ADX: {str(e)}")
            return self

    def add_obv(self) -> 'FeatureEngineer':
        """
        Add On-Balance Volume

        Returns:
            Self for method chaining
        """
        try:
            obv = [0]
            for i in range(1, len(self.df)):
                if self.df['close'].iloc[i] > self.df['close'].iloc[i - 1]:
                    obv.append(obv[-1] + self.df['volume'].iloc[i])
                elif self.df['close'].iloc[i] < self.df['close'].iloc[i - 1]:
                    obv.append(obv[-1] - self.df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])

            self.df['OBV'] = obv

            # OBV moving average
            self.df['OBV_SMA_20'] = self.df['OBV'].rolling(window=20).mean()

            logger.info("Added OBV")
            return self

        except Exception as e:
            logger.error(f"Error adding OBV: {str(e)}")
            return self

    def add_price_features(self) -> 'FeatureEngineer':
        """
        Add price-based features

        Returns:
            Self for method chaining
        """
        try:
            # Price changes
            self.df['price_change'] = self.df['close'].diff()
            self.df['price_change_pct'] = self.df['close'].pct_change() * 100

            # High-Low range
            self.df['hl_range'] = self.df['high'] - self.df['low']
            self.df['hl_range_pct'] = (self.df['hl_range'] / self.df['close']) * 100

            # Intraday momentum
            self.df['intraday_momentum'] = (self.df['close'] - self.df['open']) / self.df['open']

            # Gap (for stocks)
            self.df['gap'] = self.df['open'] - self.df['close'].shift(1)
            self.df['gap_pct'] = (self.df['gap'] / self.df['close'].shift(1)) * 100

            logger.info("Added price features")
            return self

        except Exception as e:
            logger.error(f"Error adding price features: {str(e)}")
            return self

    def add_volume_features(self) -> 'FeatureEngineer':
        """
        Add volume-based features

        Returns:
            Self for method chaining
        """
        try:
            # Volume moving averages
            self.df['volume_SMA_20'] = self.df['volume'].rolling(window=20).mean()
            self.df['volume_ratio'] = self.df['volume'] / self.df['volume_SMA_20']

            # Volume changes
            self.df['volume_change_pct'] = self.df['volume'].pct_change() * 100

            logger.info("Added volume features")
            return self

        except Exception as e:
            logger.error(f"Error adding volume features: {str(e)}")
            return self

    def add_time_features(self) -> 'FeatureEngineer':
        """
        Add time-based features (if timestamp column exists)

        Returns:
            Self for method chaining
        """
        try:
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

                self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
                self.df['month'] = self.df['timestamp'].dt.month
                self.df['quarter'] = self.df['timestamp'].dt.quarter
                self.df['is_month_start'] = self.df['timestamp'].dt.is_month_start.astype(int)
                self.df['is_month_end'] = self.df['timestamp'].dt.is_month_end.astype(int)

                logger.info("Added time features")

            return self

        except Exception as e:
            logger.error(f"Error adding time features: {str(e)}")
            return self

    def add_all_indicators(self) -> 'FeatureEngineer':
        """
        Add comprehensive suite of technical indicators

        Returns:
            Self for method chaining
        """
        logger.info("Adding all technical indicators...")

        self.add_moving_averages()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_rsi()
        self.add_stochastic_oscillator()
        self.add_atr()
        self.add_adx()
        self.add_obv()
        self.add_price_features()
        self.add_volume_features()
        self.add_time_features()

        logger.info("All indicators added successfully")
        return self

    def get_features(self, drop_na: bool = True) -> pd.DataFrame:
        """
        Get DataFrame with engineered features

        Args:
            drop_na: Whether to drop rows with NaN values

        Returns:
            DataFrame with features
        """
        if drop_na:
            return self.df.dropna()
        return self.df

    def get_feature_columns(self) -> List[str]:
        """
        Get list of all feature columns (excluding original OHLCV)

        Returns:
            List of feature column names
        """
        base_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in self.df.columns if col not in base_cols]


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG
    from ml_trading_bot.data.ingestion import StockDataFetcher

    logging.config.dictConfig(LOGGING_CONFIG)

    # Fetch sample data
    fetcher = StockDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', '2023-01-01', '2024-01-01')

    # Engineer features
    engineer = FeatureEngineer(df)
    features_df = engineer.add_all_indicators().get_features()

    print(f"\nOriginal columns: {df.columns.tolist()}")
    print(f"\nFeature columns: {engineer.get_feature_columns()}")
    print(f"\nData shape: {features_df.shape}")
    print(f"\nSample features:")
    print(features_df[['close', 'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower']].tail())
