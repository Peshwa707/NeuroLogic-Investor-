# Examples

This directory contains example scripts demonstrating various features of the ML Trading Bot.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

Complete workflow demonstration including:
- Data fetching
- Feature engineering
- Model training
- Long-term forecasting
- Day trading signal generation

**Run:**
```bash
python examples/basic_usage.py
```

### 2. Continuous Learning Demo (`continuous_learning_demo.py`)

Demonstrates the continuous learning capabilities:
- Sliding window training
- Automatic retraining
- Performance monitoring
- Model drift detection

**Run:**
```bash
python examples/continuous_learning_demo.py
```

## Notes

- All examples are self-contained and can run independently
- Make sure to install all dependencies first: `pip install -r requirements.txt`
- Examples use small datasets and fewer epochs for quick demonstration
- For production use, adjust parameters in `ml_trading_bot/config/settings.py`

## Creating Your Own Examples

You can use these examples as templates for your own implementations:

1. Copy an example file
2. Modify parameters and symbols
3. Add your custom logic
4. Run and iterate

For more detailed information, see the main README.md in the project root.
