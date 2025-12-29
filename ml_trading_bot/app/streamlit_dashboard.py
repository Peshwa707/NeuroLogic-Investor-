"""
Streamlit dashboard for ML Trading Bot
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_trading_bot.data.ingestion import StockDataFetcher, CryptoDataFetcher
from ml_trading_bot.features import FeatureEngineer
from ml_trading_bot.data.preprocessing import DataPreprocessor
from ml_trading_bot.models import LSTMModelBuilder, DayTradingClassifier
from ml_trading_bot.trading import LongTermForecaster, DayTradingSignalGenerator


# Page config
st.set_page_config(
    page_title="ML Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .signal-buy {
        color: #00cc00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .signal-sell {
        color: #cc0000;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .signal-hold {
        color: #cccc00;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data(symbol: str, market_type: str, start_date: str, end_date: str):
    """Load and cache market data"""
    if market_type == "Stock":
        fetcher = StockDataFetcher()
        return fetcher.fetch_historical_data(symbol, start_date, end_date)
    else:
        fetcher = CryptoDataFetcher()
        return fetcher.get_historical_klines(symbol, '1d', start_date, end_date)


def plot_price_chart(df: pd.DataFrame, title: str = "Price Chart"):
    """Create candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    )])

    fig.update_layout(
        title=title,
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=500
    )

    return fig


def plot_indicators(df: pd.DataFrame):
    """Plot technical indicators"""
    fig = go.Figure()

    # Price and moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        name='Close', line=dict(color='black', width=2)
    ))

    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            name='SMA 20', line=dict(color='blue', width=1)
        ))

    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            name='SMA 50', line=dict(color='red', width=1)
        ))

    if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_upper'],
            name='BB Upper', line=dict(color='gray', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_lower'],
            name='BB Lower', line=dict(color='gray', dash='dash')
        ))

    fig.update_layout(
        title="Price & Moving Averages",
        yaxis_title='Price ($)',
        template='plotly_white',
        height=400
    )

    return fig


def plot_rsi(df: pd.DataFrame):
    """Plot RSI indicator"""
    if 'RSI' not in df.columns:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        name='RSI', line=dict(color='purple', width=2)
    ))

    # Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

    fig.update_layout(
        title="RSI (Relative Strength Index)",
        yaxis_title='RSI',
        template='plotly_white',
        height=300
    )

    return fig


def plot_macd(df: pd.DataFrame):
    """Plot MACD indicator"""
    if 'MACD' not in df.columns:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        name='MACD', line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_signal'],
        name='Signal', line=dict(color='red', width=2)
    ))

    fig.add_trace(go.Bar(
        x=df.index, y=df['MACD_histogram'],
        name='Histogram', marker_color='gray'
    ))

    fig.update_layout(
        title="MACD (Moving Average Convergence Divergence)",
        yaxis_title='MACD',
        template='plotly_white',
        height=300
    )

    return fig


def main():
    """Main dashboard function"""

    # Header
    st.markdown('<div class="main-header">📈 ML Trading Bot Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    market_type = st.sidebar.selectbox("Market Type", ["Stock", "Crypto"])

    if market_type == "Stock":
        symbol = st.sidebar.text_input("Symbol", "AAPL")
        default_symbol = symbol
    else:
        symbol = st.sidebar.text_input("Symbol", "BTCUSDT")
        default_symbol = symbol

    mode = st.sidebar.selectbox("Trading Mode", ["Long-Term Forecast", "Day Trading Signals", "Both"])

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    st.sidebar.markdown("### Date Range")
    start_date = st.sidebar.date_input("Start Date", start_date)
    end_date = st.sidebar.date_input("End Date", end_date)

    # Load data button
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner(f"Loading {symbol} data..."):
            try:
                df = load_data(
                    symbol,
                    market_type,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

                if df.empty:
                    st.error(f"No data found for {symbol}")
                    return

                # Store in session state
                st.session_state['data'] = df
                st.session_state['symbol'] = symbol
                st.success(f"Loaded {len(df)} data points for {symbol}")

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return

    # Main content
    if 'data' not in st.session_state:
        st.info("👈 Configure settings in the sidebar and click 'Load Data' to begin")
        return

    df = st.session_state['data']
    symbol = st.session_state['symbol']

    # Current price
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

    with col2:
        st.metric("High (24h)", f"${df['high'].iloc[-1]:.2f}")

    with col3:
        st.metric("Low (24h)", f"${df['low'].iloc[-1]:.2f}")

    with col4:
        st.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}")

    # Chart
    st.markdown("### 📊 Price Chart")
    chart = plot_price_chart(df, f"{symbol} Price Chart")
    st.plotly_chart(chart, use_container_width=True)

    # Feature engineering
    with st.spinner("Engineering features..."):
        engineer = FeatureEngineer(df)
        features_df = engineer.add_all_indicators().get_features()

    # Technical Indicators
    st.markdown("### 📈 Technical Indicators")

    col1, col2 = st.columns(2)

    with col1:
        indicators_chart = plot_indicators(features_df)
        st.plotly_chart(indicators_chart, use_container_width=True)

        macd_chart = plot_macd(features_df)
        if macd_chart:
            st.plotly_chart(macd_chart, use_container_width=True)

    with col2:
        rsi_chart = plot_rsi(features_df)
        if rsi_chart:
            st.plotly_chart(rsi_chart, use_container_width=True)

        # Current indicator values
        st.markdown("#### Current Values")
        latest = features_df.iloc[-1]

        indicator_cols = st.columns(2)
        with indicator_cols[0]:
            st.metric("RSI", f"{latest.get('RSI', 0):.2f}")
            st.metric("MACD", f"{latest.get('MACD', 0):.4f}")

        with indicator_cols[1]:
            st.metric("SMA 20", f"${latest.get('SMA_20', 0):.2f}")
            st.metric("SMA 50", f"${latest.get('SMA_50', 0):.2f}")

    # Predictions
    st.markdown("---")
    st.markdown("### 🤖 AI Predictions")

    if mode in ["Long-Term Forecast", "Both"]:
        st.markdown("#### 📅 5-Day Forecast")

        if st.button("Generate Long-Term Forecast"):
            with st.spinner("Training model and generating forecast..."):
                try:
                    # Prepare data
                    preprocessor = DataPreprocessor(lookback=60)
                    feature_cols = ['close', 'volume', 'RSI', 'MACD', 'SMA_20']
                    X, y = preprocessor.prepare_multivariate_data(
                        features_df, target_col='close', feature_cols=feature_cols
                    )

                    # Train model (simple version for demo)
                    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, train_ratio=0.8)
                    model_builder = LSTMModelBuilder(lookback=60, n_features=len(feature_cols), units=50)
                    model = model_builder.create_multivariate_lstm()
                    model_builder.train(X_train[:80], y_train[:80], X_test[:20], y_test[:20], epochs=10, batch_size=32)

                    # Generate forecast
                    forecaster = LongTermForecaster(model_builder, preprocessor, forecast_days=5)
                    forecast = forecaster.predict_trend(features_df)

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Trend", forecast['trend'])

                    with col2:
                        st.metric("Expected Change", f"{forecast['expected_change_pct']:.2f}%")

                    with col3:
                        st.metric("Confidence", f"{forecast['confidence']:.2%}")

                    # Forecast table
                    forecast_df = pd.DataFrame({
                        'Date': forecast['forecast_dates'],
                        'Predicted Price': [f"${p:.2f}" for p in forecast['predictions']]
                    })

                    st.dataframe(forecast_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")

    if mode in ["Day Trading Signals", "Both"]:
        st.markdown("#### ⚡ Day Trading Signal")

        if st.button("Generate Trading Signal"):
            with st.spinner("Analyzing market and generating signal..."):
                try:
                    # Train classifier (simplified for demo)
                    classifier = DayTradingClassifier(threshold=0.005)
                    features_df_copy = features_df.copy()
                    features_df_copy['target'] = classifier.create_target(features_df_copy)
                    X, y = classifier.prepare_features(features_df_copy)
                    classifier.train(X, y, test_size=0.2)

                    # Generate signal
                    signal_gen = DayTradingSignalGenerator(classifier, engineer)
                    signal = signal_gen.generate_signal(df.tail(100))

                    # Display signal
                    signal_class = f"signal-{signal['signal'].lower()}"
                    st.markdown(f"<div class='{signal_class}'>{signal['signal']}</div>", unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Confidence", f"{signal['confidence']:.2%}")
                        st.metric("Signal Strength", signal['metadata']['signal_strength'])

                    with col2:
                        st.metric("Current Price", f"${signal['price']:.2f}")

                    # Technical indicators that led to signal
                    st.markdown("**Supporting Indicators:**")
                    tech_indicators = signal['metadata']['technical_indicators']

                    ind_cols = st.columns(3)
                    for i, (indicator, value) in enumerate(tech_indicators.items()):
                        with ind_cols[i % 3]:
                            st.write(f"**{indicator}:** {value:.2f}")

                except Exception as e:
                    st.error(f"Error generating signal: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "ML Trading Bot v1.0 | For educational purposes only"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
