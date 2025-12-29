"""
Basic usage example for ML Trading Bot
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_trading_bot.data.ingestion import StockDataFetcher
from ml_trading_bot.features import FeatureEngineer
from ml_trading_bot.data.preprocessing import DataPreprocessor
from ml_trading_bot.models import LSTMModelBuilder, DayTradingClassifier
from ml_trading_bot.trading import LongTermForecaster, DayTradingSignalGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main example function"""
    print("=" * 60)
    print("ML Trading Bot - Basic Usage Example")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/6] Fetching stock data...")
    fetcher = StockDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', '2023-01-01', '2024-01-01')
    print(f"✓ Loaded {len(df)} data points")

    # 2. Engineer features
    print("\n[2/6] Engineering features...")
    engineer = FeatureEngineer(df)
    features_df = engineer.add_all_indicators().get_features()
    print(f"✓ Generated {len(engineer.get_feature_columns())} features")
    print(f"  Sample features: {engineer.get_feature_columns()[:5]}")

    # 3. Prepare data for ML
    print("\n[3/6] Preprocessing data...")
    preprocessor = DataPreprocessor(lookback=60)
    feature_cols = ['close', 'volume', 'RSI', 'MACD', 'SMA_20', 'BB_upper', 'BB_lower']
    X, y = preprocessor.prepare_multivariate_data(features_df, target_col='close', feature_cols=feature_cols)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, train_ratio=0.8)
    X_train, X_val, y_train, y_val = preprocessor.train_test_split(X_train, y_train, train_ratio=0.8)
    print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 4. Train LSTM model
    print("\n[4/6] Training LSTM model...")
    model_builder = LSTMModelBuilder(lookback=60, n_features=len(feature_cols), units=50, dropout_rate=0.2)
    model = model_builder.create_multivariate_lstm()
    print(f"✓ Model created with {model.count_params():,} parameters")

    print("  Training for 10 epochs (this may take a few minutes)...")
    history = model_builder.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    print(f"✓ Training completed")

    # 5. Generate long-term forecast
    print("\n[5/6] Generating 5-day forecast...")
    forecaster = LongTermForecaster(model_builder, preprocessor, forecast_days=5)
    forecast = forecaster.predict_trend(features_df)
    print(f"✓ Forecast generated")
    print(f"  Current Price: ${forecast['current_price']:.2f}")
    print(f"  Trend: {forecast['trend']}")
    print(f"  Expected Change: {forecast['expected_change_pct']:.2f}%")
    print(f"  5-Day Prediction: ${forecast['final_price']:.2f}")
    print(f"  Confidence: {forecast['confidence']:.2%}")

    # 6. Generate day trading signal
    print("\n[6/6] Generating day trading signal...")
    classifier = DayTradingClassifier(threshold=0.005)
    features_df_copy = features_df.copy()
    features_df_copy['target'] = classifier.create_target(features_df_copy)
    X_clf, y_clf = classifier.prepare_features(features_df_copy)
    classifier.train(X_clf, y_clf, test_size=0.2)

    signal_gen = DayTradingSignalGenerator(classifier, engineer)
    signal = signal_gen.generate_signal(df.tail(100))
    print(f"✓ Signal generated")
    print(f"  Signal: {signal['signal']}")
    print(f"  Confidence: {signal['confidence']:.2%}")
    print(f"  Signal Strength: {signal['metadata']['signal_strength']}")
    print(f"  Current Price: ${signal['price']:.2f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
