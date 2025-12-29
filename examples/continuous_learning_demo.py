"""
Continuous learning demonstration
"""
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_trading_bot.data.ingestion import StockDataFetcher
from ml_trading_bot.features import FeatureEngineer
from ml_trading_bot.data.preprocessing import DataPreprocessor
from ml_trading_bot.models import LSTMModelBuilder
from ml_trading_bot.continuous_learning import ContinuousLearningManager, ModelDriftDetector, SlidingWindowTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function"""
    print("=" * 70)
    print("ML Trading Bot - Continuous Learning Demonstration")
    print("=" * 70)

    # Fetch data
    print("\n[1/4] Fetching data...")
    fetcher = StockDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', '2022-01-01', '2024-01-01')
    print(f"✓ Loaded {len(df)} data points")

    # Engineer features
    print("\n[2/4] Engineering features...")
    engineer = FeatureEngineer(df)
    features_df = engineer.add_all_indicators().get_features()
    print(f"✓ Features engineered")

    # Prepare data
    preprocessor = DataPreprocessor(lookback=60)
    feature_cols = ['close', 'volume', 'RSI', 'MACD', 'SMA_20']

    # Define model factory
    def model_factory():
        return LSTMModelBuilder(
            lookback=60,
            n_features=len(feature_cols),
            units=50,
            dropout_rate=0.2
        )

    # Sliding window training
    print("\n[3/4] Demonstrating sliding window training...")
    trainer = SlidingWindowTrainer(
        model_factory=model_factory,
        window_size=90,
        retrain_frequency=5
    )

    # Initial training
    print("  Training initial model...")
    metrics = trainer.train_on_window(features_df, preprocessor)
    if 'error' not in metrics:
        print(f"  ✓ Initial training completed")
        print(f"    Train samples: {metrics['train_samples']}")
        print(f"    Val samples: {metrics['val_samples']}")
    else:
        print(f"  ✗ Training failed: {metrics['error']}")
        return

    # Simulate daily updates
    print("\n  Simulating 30 days of operation...")
    retraining_days = []

    for day in range(30):
        retrained, metrics = trainer.update_and_check_retrain(features_df, preprocessor)

        if retrained:
            retraining_days.append(day + 1)
            print(f"  Day {day+1}: ✓ Model retrained")
        elif (day + 1) % 10 == 0:
            print(f"  Day {day+1}: No retraining needed (days since retrain: {trainer.days_since_retrain})")

    print(f"\n  Summary:")
    print(f"    Total retrainings: {len(retraining_days)}")
    print(f"    Retraining days: {retraining_days}")

    # Continuous learning manager
    print("\n[4/4] Demonstrating continuous learning manager...")

    model = trainer.get_model()
    if model is None:
        print("  ✗ Model not available")
        return

    manager = ContinuousLearningManager(model, error_threshold=0.05, window_size=50)

    # Simulate predictions
    print("  Simulating 100 predictions...")
    np.random.seed(42)

    for i in range(100):
        # Simulate actual price
        actual = 150 + np.random.randn() * 5

        # Simulate prediction with increasing error
        prediction = actual + np.random.randn() * 2 + (i * 0.05)

        manager.add_prediction(prediction, actual)

        if (i + 1) % 25 == 0:
            metrics = manager.get_current_metrics()
            should_retrain = manager.should_retrain()

            print(f"\n  After {i+1} predictions:")
            print(f"    SMAPE: {metrics['smape']:.2f}%")
            print(f"    MAPE: {metrics['mape']:.2f}%")
            print(f"    Directional Accuracy: {metrics['directional_accuracy']:.2%}")
            print(f"    Should retrain: {'Yes ⚠️' if should_retrain else 'No ✓'}")

            manager.save_metrics_history()

    # Drift detection
    print("\n  Model Drift Detection:")
    baseline_metrics = {
        'smape': 2.0,
        'directional_accuracy': 0.75
    }

    drift_detector = ModelDriftDetector(baseline_metrics, alert_threshold=0.2)
    current_metrics = manager.get_current_metrics()
    drift_result = drift_detector.detect_drift(current_metrics)

    print(f"    Drift detected: {'Yes ⚠️' if drift_result['drift_detected'] else 'No ✓'}")

    if drift_result['degraded_metrics']:
        print("    Degraded metrics:")
        for metric in drift_result['degraded_metrics']:
            print(f"      - {metric['metric']}: {metric['baseline']:.4f} -> "
                  f"{metric['current']:.4f} ({metric['change']:+.2%})")

    print("\n" + "=" * 70)
    print("Continuous learning demonstration completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
