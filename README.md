# 📈 ML Trading Bot: Continuously Learning Stock & Crypto Prediction System

A production-grade, continuously learning stock and cryptocurrency prediction bot built with Python, TensorFlow, and modern MLOps practices. The system supports both **long-term trend forecasting** and **short-term day trading signals**, with the ability to learn from live market data and improve predictions over time.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

### Core Capabilities
- ✅ **Dual Trading Modes**: Long-term forecasting (5+ days) and day trading signals
- ✅ **Continuous Learning**: Automatic model retraining based on performance metrics
- ✅ **Multi-Asset Support**: Stocks (Yahoo Finance) and cryptocurrencies (Binance)
- ✅ **Event-Driven Architecture**: Apache Kafka for real-time data streaming
- ✅ **Time-Series Storage**: TimescaleDB for efficient historical data management
- ✅ **Real-Time Dashboard**: Interactive Streamlit interface with live predictions
- ✅ **Technical Indicators**: 20+ indicators including RSI, MACD, Bollinger Bands, etc.
- ✅ **Sentiment Analysis**: VADER and FinBERT for news sentiment integration

### Machine Learning Models
- **LSTM Networks**: Univariate, multivariate, bidirectional, and stateful variants
- **XGBoost Classifier**: For binary day trading signals (BUY/SELL/HOLD)
- **Random Forest**: Alternative classification model
- **Online Learning**: River library for real-time model updates

### Advanced Features
- **Model Drift Detection**: Automatic detection of performance degradation
- **Sliding Window Training**: Continuous retraining on recent data windows
- **Adaptive Window Sizing**: Dynamic adjustment based on model performance
- **Feature Engineering**: Automated technical indicator generation
- **Risk Management**: Position sizing and signal validation

## 🚂 Quick Deploy to Railway (Recommended)

**Deploy to the cloud in 3 minutes - No setup required!**

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

1. Click the button above
2. Wait for deployment (2-5 minutes)
3. Open your app and configure API keys through the web interface
4. Start predicting!

📖 **Detailed Railway Guide**: [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md)
📖 **Quick Start Guide**: [README_RAILWAY.md](README_RAILWAY.md)

**Why Railway?**
- ✅ No Docker or Kubernetes setup
- ✅ Free tier available ($5 credit/month)
- ✅ Configure API keys through web interface
- ✅ Automatic HTTPS and custom domains
- ✅ Built-in monitoring and logs

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingestion │───▶│  Model Training │───▶│  Live Prediction│
│     Module      │    │     Service     │    │     Service     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         └──────────────────────┴──────────────────────┘
                    Apache Kafka Message Queue
                              │
                    ┌─────────┴─────────┐
                    │   TimescaleDB     │
                    │  Time-Series DB   │
                    └───────────────────┘
```

## 📁 Project Structure

```
ml_trading_bot/
├── config/
│   ├── settings.py              # Configuration settings
│   └── credentials.py           # API keys (gitignored)
├── data/
│   ├── ingestion/
│   │   ├── stock_fetcher.py    # Yahoo Finance integration
│   │   ├── crypto_fetcher.py   # Binance API integration
│   │   └── __init__.py
│   ├── storage/
│   │   ├── timescale_db.py     # TimescaleDB interface
│   │   ├── kafka_manager.py    # Kafka producer/consumer
│   │   └── __init__.py
│   └── preprocessing/
│       ├── data_preprocessor.py # Feature scaling and sequences
│       └── __init__.py
├── features/
│   ├── technical_indicators.py  # 20+ technical indicators
│   ├── sentiment_analyzer.py    # VADER & FinBERT sentiment
│   └── __init__.py
├── models/
│   ├── lstm_models.py          # LSTM architectures
│   ├── classification_models.py # XGBoost & Random Forest
│   ├── online_learning.py      # River online learning
│   └── __init__.py
├── continuous_learning/
│   ├── feedback_loop.py        # Performance tracking
│   ├── sliding_window_trainer.py # Retraining system
│   └── __init__.py
├── trading/
│   ├── long_term_forecaster.py # Multi-day predictions
│   ├── signal_generator.py     # Day trading signals
│   └── __init__.py
├── app/
│   └── streamlit_dashboard.py  # Interactive dashboard
├── saved_models/               # Trained model storage
├── logs/                       # Application logs
├── tests/                      # Unit tests
├── docker-compose.yml          # Infrastructure setup
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Two Deployment Options

#### Option 1: Railway (Cloud - Recommended for beginners)
See [Railway Deployment Guide](RAILWAY_DEPLOYMENT.md) or [Quick Start](README_RAILWAY.md)

#### Option 2: Local Development (Advanced)

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (for infrastructure)
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd NeuroLogic-Investor-
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

Required API keys:
- `ALPHA_VANTAGE_API_KEY`: Get from [Alpha Vantage](https://www.alphavantage.co/)
- `BINANCE_API_KEY` & `BINANCE_API_SECRET`: Get from [Binance](https://www.binance.com/en/my/settings/api-management)
- `NEWS_API_KEY`: Get from [NewsAPI](https://newsapi.org/)

### 4. Start Infrastructure Services

```bash
# Start TimescaleDB, Kafka, and other services
docker-compose up -d

# Verify services are running
docker-compose ps
```

Services will be available at:
- TimescaleDB: `localhost:5432`
- Kafka: `localhost:9092`
- Kafka UI: `http://localhost:8080`
- Jupyter: `http://localhost:8888`

### 5. Initialize Database

```bash
python -c "
from ml_trading_bot.data.storage import TimeSeriesDB
from ml_trading_bot.config.settings import TIMESCALE_CONFIG

db = TimeSeriesDB(**TIMESCALE_CONFIG)
db.create_tables()
print('Database initialized successfully!')
"
```

### 6. Launch Dashboard

```bash
streamlit run ml_trading_bot/app/streamlit_dashboard.py
```

The dashboard will open at `http://localhost:8501`

## 📊 Usage Examples

### Fetch Historical Data

```python
from ml_trading_bot.data.ingestion import StockDataFetcher

# Initialize fetcher
fetcher = StockDataFetcher()

# Fetch stock data
df = fetcher.fetch_historical_data('AAPL', '2023-01-01', '2024-01-01')
print(df.head())

# Fetch real-time quote
quote = fetcher.fetch_realtime_quote('AAPL')
print(f"Current price: ${quote['current_price']}")
```

### Engineer Features

```python
from ml_trading_bot.features import FeatureEngineer

# Create feature engineer
engineer = FeatureEngineer(df)

# Add all indicators
features_df = engineer.add_all_indicators().get_features()

# Or add specific indicators
features_df = (engineer
    .add_moving_averages([20, 50, 200])
    .add_rsi()
    .add_macd()
    .add_bollinger_bands()
    .get_features())

print(features_df.columns.tolist())
```

### Train LSTM Model

```python
from ml_trading_bot.models import LSTMModelBuilder
from ml_trading_bot.data.preprocessing import DataPreprocessor

# Prepare data
preprocessor = DataPreprocessor(lookback=60)
feature_cols = ['close', 'volume', 'RSI', 'MACD', 'SMA_20']
X, y = preprocessor.prepare_multivariate_data(features_df, target_col='close', feature_cols=feature_cols)

# Split data
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, train_ratio=0.8)
X_train, X_val, y_train, y_val = preprocessor.train_test_split(X_train, y_train, train_ratio=0.8)

# Create and train model
model_builder = LSTMModelBuilder(lookback=60, n_features=len(feature_cols), units=100)
model = model_builder.create_multivariate_lstm()

# Train
history = model_builder.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

# Save model
model_builder.save_model('saved_models/lstm_model.h5')
```

### Generate Long-Term Forecast

```python
from ml_trading_bot.trading import LongTermForecaster

# Create forecaster
forecaster = LongTermForecaster(model_builder, preprocessor, forecast_days=5)

# Generate forecast
forecast = forecaster.predict_trend(features_df)

print(f"Current Price: ${forecast['current_price']:.2f}")
print(f"Trend: {forecast['trend']}")
print(f"Expected Change: {forecast['expected_change_pct']:.2f}%")
print(f"5-Day Prediction: ${forecast['final_price']:.2f}")
```

### Generate Day Trading Signals

```python
from ml_trading_bot.models import DayTradingClassifier
from ml_trading_bot.trading import DayTradingSignalGenerator

# Train classifier
classifier = DayTradingClassifier(threshold=0.005)
features_df['target'] = classifier.create_target(features_df)
X, y = classifier.prepare_features(features_df)
classifier.train(X, y)

# Create signal generator
signal_gen = DayTradingSignalGenerator(classifier, engineer)

# Generate signal
signal = signal_gen.generate_signal(df.tail(100))

print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']:.2%}")
print(f"Signal Strength: {signal['metadata']['signal_strength']}")
```

### Continuous Learning

```python
from ml_trading_bot.continuous_learning import ContinuousLearningManager

# Create continuous learning manager
manager = ContinuousLearningManager(model, error_threshold=0.05, window_size=50)

# Add predictions and actuals
for prediction, actual in zip(predictions, actuals):
    manager.add_prediction(prediction, actual)

# Check if retraining is needed
if manager.should_retrain():
    print("Model performance degraded, retraining recommended")
    manager.retrain(X_new, y_new)

# Get performance metrics
metrics = manager.get_current_metrics()
print(f"SMAPE: {metrics['smape']:.2f}%")
print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
```

### Online Learning

```python
from ml_trading_bot.models import OnlineLearner

# Create online learner
learner = OnlineLearner(model_type='logistic')

# Stream data and learn
for features, target in data_stream:
    # Learn from new data
    learner.learn_one(features, target)

    # Make prediction
    signal = learner.predict_signal(features)

# Get performance
performance = learner.get_performance()
print(f"Accuracy: {performance['accuracy']:.2%}")
print(f"ROC-AUC: {performance['roc_auc']:.2%}")
```

## 📈 Performance Metrics

The system tracks multiple performance metrics:

- **SMAPE** (Symmetric Mean Absolute Percentage Error): For price predictions
- **MAPE** (Mean Absolute Percentage Error): Alternative error metric
- **Directional Accuracy**: Percentage of correct trend predictions
- **ROC-AUC**: For classification models
- **Sharpe Ratio**: For trading performance (in backtesting)
- **Maximum Drawdown**: Risk metric

## 🔧 Configuration

### Model Configuration (`ml_trading_bot/config/settings.py`)

```python
MODEL_CONFIG = {
    'lookback_period': 60,        # Days to look back
    'forecast_days': 5,           # Days to forecast
    'batch_size': 32,
    'epochs': 50,
    'lstm_units': 100,
    'dropout_rate': 0.2
}

TRAINING_CONFIG = {
    'sliding_window_size': 60,    # Days
    'retrain_frequency': 5,       # Days
    'error_threshold': 0.05       # 5% error threshold
}

DAY_TRADING_CONFIG = {
    'price_change_threshold': 0.005,  # 0.5%
    'buy_confidence_threshold': 0.6,
    'sell_confidence_threshold': 0.4
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_data_ingestion.py

# Run with coverage
pytest --cov=ml_trading_bot tests/
```

## 📝 Logging

Logs are stored in `ml_trading_bot/logs/trading_bot.log`. Configure logging in `ml_trading_bot/config/settings.py`:

```python
LOGGING_CONFIG = {
    'version': 1,
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'trading_bot.log',
            'level': 'INFO'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO'
        }
    }
}
```

## ⚠️ Important Notes

### Risk Disclaimer
**This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.**

### Best Practices
1. **Never hardcode API keys** - Always use environment variables
2. **Start with paper trading** - Test thoroughly before live trading
3. **Monitor performance** - Track metrics and retrain when needed
4. **Implement risk management** - Use stop-losses and position sizing
5. **Backtest thoroughly** - Validate strategies on historical data

### Security
- API keys are stored in `.env` (not committed to Git)
- Use encrypted connections for production
- Implement proper authentication for APIs
- Regularly rotate API keys

## 🛠️ Troubleshooting

### Common Issues

**Issue**: "Cannot connect to TimescaleDB"
```bash
# Solution: Ensure Docker containers are running
docker-compose ps
docker-compose up -d timescaledb
```

**Issue**: "Module not found"
```bash
# Solution: Ensure you're in the virtual environment and dependencies are installed
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Issue**: "API rate limit exceeded"
```bash
# Solution: Implement caching or reduce request frequency
# Check API documentation for rate limits
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras**: Deep learning framework
- **XGBoost**: Gradient boosting library
- **River**: Online machine learning
- **Yahoo Finance (yfinance)**: Stock data
- **Binance**: Cryptocurrency data
- **pandas-ta**: Technical analysis library
- **Streamlit**: Dashboard framework

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review existing issues and discussions

## 🗺️ Roadmap

- [ ] Add more ML models (Transformers, GRU, etc.)
- [ ] Implement backtesting engine
- [ ] Add more exchanges (Coinbase, Kraken, etc.)
- [ ] Portfolio optimization algorithms
- [ ] Automated trading execution
- [ ] Advanced risk management tools
- [ ] Mobile app
- [ ] Docker deployment guides
- [ ] Kubernetes orchestration

---

**Built with ❤️ for the trading community**

*Last Updated: 2024*
