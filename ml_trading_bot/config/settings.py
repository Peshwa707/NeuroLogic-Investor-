"""
Configuration settings for ML Trading Bot
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAVED_MODELS_DIR = BASE_DIR / "ml_trading_bot" / "saved_models"
LOGS_DIR = BASE_DIR / "ml_trading_bot" / "logs"

# Create directories if they don't exist
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration
TIMESCALE_CONFIG = {
    'host': os.getenv('TIMESCALE_HOST', 'localhost'),
    'port': os.getenv('TIMESCALE_PORT', 5432),
    'database': os.getenv('TIMESCALE_DB', 'trading_bot'),
    'user': os.getenv('TIMESCALE_USER', 'postgres'),
    'password': os.getenv('TIMESCALE_PASSWORD', 'postgres')
}

# Kafka Configuration
KAFKA_CONFIG = {
    'bootstrap_servers': os.getenv('KAFKA_SERVERS', 'localhost:9092'),
    'topics': {
        'stock_prices': 'stock-prices',
        'crypto_prices': 'crypto-prices',
        'predictions': 'predictions',
        'signals': 'trading-signals'
    }
}

# Model Configuration
MODEL_CONFIG = {
    'lookback_period': 60,  # Number of days to look back
    'forecast_days': 5,  # Number of days to forecast
    'batch_size': 32,
    'epochs': 50,
    'lstm_units': 100,
    'dropout_rate': 0.2,
    'learning_rate': 0.001
}

# Training Configuration
TRAINING_CONFIG = {
    'sliding_window_size': 60,  # Days
    'retrain_frequency': 5,  # Days
    'error_threshold': 0.05,  # 5% error threshold for retraining
    'validation_split': 0.2
}

# Day Trading Configuration
DAY_TRADING_CONFIG = {
    'price_change_threshold': 0.005,  # 0.5% price change
    'buy_confidence_threshold': 0.6,
    'sell_confidence_threshold': 0.4
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'ma_windows': [5, 10, 20, 50, 200],
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_window': 20,
    'bollinger_std': 2,
    'rsi_window': 14
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'trading_bot.log',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO',
    },
}
