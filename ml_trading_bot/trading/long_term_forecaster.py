"""
Long-term forecaster for multi-day price prediction
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class LongTermForecaster:
    """Generate multi-day price forecasts using LSTM models"""

    def __init__(self, model, preprocessor, forecast_days: int = 5):
        """
        Initialize long-term forecaster

        Args:
            model: Trained LSTM model (LSTMModelBuilder instance)
            preprocessor: DataPreprocessor instance
            forecast_days: Number of days to forecast
        """
        self.model = model
        self.preprocessor = preprocessor
        self.forecast_days = forecast_days

        logger.info(f"LongTermForecaster initialized with forecast_days={forecast_days}")

    def predict_trend(self, recent_data: pd.DataFrame) -> dict:
        """
        Generate multi-day price forecast

        Args:
            recent_data: Recent historical data

        Returns:
            Dictionary with forecast results
        """
        try:
            logger.info(f"Generating {self.forecast_days}-day forecast...")

            # Prepare data
            X, _ = self.preprocessor.prepare_multivariate_data(recent_data)

            if len(X) == 0:
                logger.error("No sequences generated from recent data")
                return self._empty_forecast()

            # Get last sequence
            last_sequence = X[-1:]

            # Make predictions
            predictions = []
            current_sequence = last_sequence.copy()

            for day in range(self.forecast_days):
                # Predict next value
                pred = self.model.predict(current_sequence)[0]
                predictions.append(pred)

                # Update sequence for next prediction (recursive forecasting)
                # Shift sequence and add new prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred  # Update close price feature

            # Inverse transform predictions to original scale
            predictions_array = np.array(predictions).reshape(-1, 1)
            predictions_original = self.preprocessor.inverse_scale(predictions_array).flatten()

            # Calculate trend metrics
            current_price = recent_data['close'].iloc[-1]
            final_price = predictions_original[-1]

            trend = 'BULLISH' if final_price > current_price else 'BEARISH'
            expected_change_pct = ((final_price / current_price) - 1) * 100

            # Generate forecast dates
            last_date = pd.to_datetime(recent_data['timestamp'].iloc[-1])
            forecast_dates = [
                (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(self.forecast_days)
            ]

            result = {
                'current_price': float(current_price),
                'predictions': predictions_original.tolist(),
                'forecast_dates': forecast_dates,
                'final_price': float(final_price),
                'trend': trend,
                'expected_change_pct': float(expected_change_pct),
                'confidence': self._calculate_confidence(predictions_original),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(
                f"Forecast complete: {trend} trend, "
                f"expected change: {expected_change_pct:.2f}%"
            )

            return result

        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return self._empty_forecast()

    def predict_single_day(self, recent_data: pd.DataFrame) -> dict:
        """
        Predict next day's price

        Args:
            recent_data: Recent historical data

        Returns:
            Dictionary with next-day prediction
        """
        try:
            # Prepare data
            X, _ = self.preprocessor.prepare_multivariate_data(recent_data)

            if len(X) == 0:
                logger.error("No sequences generated")
                return {}

            # Predict
            prediction = self.model.predict(X[-1:])[0]

            # Inverse transform
            prediction_original = self.preprocessor.inverse_scale(
                np.array([[prediction]])
            )[0][0]

            current_price = recent_data['close'].iloc[-1]
            expected_change_pct = ((prediction_original / current_price) - 1) * 100

            result = {
                'current_price': float(current_price),
                'predicted_price': float(prediction_original),
                'expected_change_pct': float(expected_change_pct),
                'direction': 'UP' if prediction_original > current_price else 'DOWN',
                'timestamp': datetime.now().isoformat()
            }

            logger.info(
                f"Next-day prediction: ${prediction_original:.2f} "
                f"({expected_change_pct:+.2f}%)"
            )

            return result

        except Exception as e:
            logger.error(f"Error predicting single day: {str(e)}")
            return {}

    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        """
        Calculate forecast confidence based on prediction variance

        Args:
            predictions: Array of predictions

        Returns:
            Confidence score (0-1)
        """
        try:
            # Lower variance in predictions = higher confidence
            variance = np.var(predictions)
            mean_price = np.mean(predictions)

            # Normalized variance (coefficient of variation)
            cv = (np.sqrt(variance) / mean_price) if mean_price != 0 else 1

            # Convert to confidence score (inverse relationship)
            confidence = 1 / (1 + cv)

            return float(np.clip(confidence, 0, 1))

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _empty_forecast(self) -> dict:
        """Return empty forecast dict"""
        return {
            'current_price': 0.0,
            'predictions': [],
            'forecast_dates': [],
            'final_price': 0.0,
            'trend': 'NEUTRAL',
            'expected_change_pct': 0.0,
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'error': 'forecast_failed'
        }

    def get_price_range(self, predictions: List[float]) -> dict:
        """
        Calculate price range from predictions

        Args:
            predictions: List of predicted prices

        Returns:
            Dictionary with min, max, and range
        """
        try:
            predictions_array = np.array(predictions)

            return {
                'min_price': float(np.min(predictions_array)),
                'max_price': float(np.max(predictions_array)),
                'price_range': float(np.max(predictions_array) - np.min(predictions_array)),
                'avg_price': float(np.mean(predictions_array))
            }

        except Exception as e:
            logger.error(f"Error calculating price range: {str(e)}")
            return {}


class TrendAnalyzer:
    """Analyze price trends and patterns"""

    @staticmethod
    def identify_support_resistance(prices: pd.Series, window: int = 20) -> dict:
        """
        Identify support and resistance levels

        Args:
            prices: Price series
            window: Window size for calculation

        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Calculate rolling min/max
            support = prices.rolling(window=window).min().iloc[-1]
            resistance = prices.rolling(window=window).max().iloc[-1]

            return {
                'support': float(support),
                'resistance': float(resistance),
                'current_position': float((prices.iloc[-1] - support) / (resistance - support))
                    if resistance != support else 0.5
            }

        except Exception as e:
            logger.error(f"Error identifying support/resistance: {str(e)}")
            return {}

    @staticmethod
    def calculate_trend_strength(prices: pd.Series, window: int = 20) -> dict:
        """
        Calculate trend strength using linear regression

        Args:
            prices: Price series
            window: Window size

        Returns:
            Dictionary with trend metrics
        """
        try:
            recent_prices = prices.tail(window).values
            x = np.arange(len(recent_prices))

            # Linear regression
            slope, intercept = np.polyfit(x, recent_prices, 1)

            # R-squared
            y_pred = slope * x + intercept
            ss_tot = np.sum((recent_prices - np.mean(recent_prices)) ** 2)
            ss_res = np.sum((recent_prices - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return {
                'slope': float(slope),
                'trend_direction': 'UP' if slope > 0 else 'DOWN',
                'trend_strength': float(abs(slope)),
                'r_squared': float(r_squared)
            }

        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return {}


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG
    from ml_trading_bot.data.ingestion import StockDataFetcher
    from ml_trading_bot.features import FeatureEngineer
    from ml_trading_bot.data.preprocessing import DataPreprocessor
    from ml_trading_bot.models import LSTMModelBuilder

    logging.config.dictConfig(LOGGING_CONFIG)

    print("=== Long-Term Forecasting Example ===\n")

    # Fetch and prepare data
    print("Fetching data...")
    fetcher = StockDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', '2023-01-01', '2024-01-01')

    print("Engineering features...")
    engineer = FeatureEngineer(df)
    features_df = engineer.add_all_indicators().get_features()

    # Prepare and train model
    print("Training model...")
    preprocessor = DataPreprocessor(lookback=60)
    feature_cols = ['close', 'volume', 'RSI', 'MACD', 'SMA_20', 'BB_upper', 'BB_lower']
    X, y = preprocessor.prepare_multivariate_data(features_df, target_col='close', feature_cols=feature_cols)

    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, train_ratio=0.8)
    X_train, X_val, y_train, y_val = preprocessor.train_test_split(X_train, y_train, train_ratio=0.8)

    model_builder = LSTMModelBuilder(lookback=60, n_features=len(feature_cols), units=50, dropout_rate=0.2)
    model = model_builder.create_multivariate_lstm()
    model_builder.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)

    # Create forecaster
    print("\nCreating forecaster...")
    forecaster = LongTermForecaster(model_builder, preprocessor, forecast_days=5)

    # Generate forecast
    print("\nGenerating 5-day forecast...")
    forecast = forecaster.predict_trend(features_df)

    print(f"\nForecast Results:")
    print(f"Current Price: ${forecast['current_price']:.2f}")
    print(f"Trend: {forecast['trend']}")
    print(f"Expected Change: {forecast['expected_change_pct']:.2f}%")
    print(f"Final Price (Day 5): ${forecast['final_price']:.2f}")
    print(f"Confidence: {forecast['confidence']:.2%}")
    print(f"\nDaily Predictions:")
    for date, price in zip(forecast['forecast_dates'], forecast['predictions']):
        print(f"  {date}: ${price:.2f}")

    # Single day prediction
    print("\nNext-day prediction...")
    next_day = forecaster.predict_single_day(features_df)
    print(f"Predicted Price: ${next_day['predicted_price']:.2f} ({next_day['expected_change_pct']:+.2f}%)")

    # Trend analysis
    print("\nTrend Analysis...")
    analyzer = TrendAnalyzer()
    support_resistance = analyzer.identify_support_resistance(features_df['close'])
    print(f"Support: ${support_resistance['support']:.2f}")
    print(f"Resistance: ${support_resistance['resistance']:.2f}")

    trend_strength = analyzer.calculate_trend_strength(features_df['close'])
    print(f"Trend Direction: {trend_strength['trend_direction']}")
    print(f"Trend Strength: {trend_strength['trend_strength']:.4f}")
    print(f"R-squared: {trend_strength['r_squared']:.4f}")
