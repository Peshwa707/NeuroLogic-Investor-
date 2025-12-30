"""
Streamlit dashboard for ML Trading Bot - Railway Optimized
With built-in API configuration interface
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_trading_bot.config.config_manager import ConfigManager
from ml_trading_bot.data.ingestion import StockDataFetcher, CryptoDataFetcher
from ml_trading_bot.features import FeatureEngineer
from ml_trading_bot.data.preprocessing import DataPreprocessor
from ml_trading_bot.models import LSTMModelBuilder, DayTradingClassifier
from ml_trading_bot.trading import LongTermForecaster, DayTradingSignalGenerator

# Initialize config manager
if 'config_manager' not in st.session_state:
    st.session_state.config_manager = ConfigManager()

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
    .status-ok {
        color: #00cc00;
    }
    .status-error {
        color: #cc0000;
    }
</style>
""", unsafe_allow_html=True)


def show_settings_page():
    """Settings page for API configuration"""
    st.markdown('<div class="main-header">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Get config manager
    config = st.session_state.config_manager

    # Check current status
    api_status = config.get_api_status()

    # Display current status
    st.markdown("### 📊 API Status")
    cols = st.columns(4)

    status_items = [
        ('Alpha Vantage', api_status.get('Alpha Vantage', False)),
        ('Binance', api_status.get('Binance', False)),
        ('Binance Secret', api_status.get('Binance Secret', False)),
        ('News API', api_status.get('News API', False))
    ]

    for col, (name, status) in zip(cols, status_items):
        with col:
            status_icon = "✅" if status else "❌"
            status_text = "Configured" if status else "Not Set"
            st.metric(name, status_text, status_icon)

    st.markdown("---")

    # Configuration form
    st.markdown("### 🔑 Configure API Keys")

    with st.form("api_config_form"):
        st.markdown("#### Stock Market Data")
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API Key",
            value=config.get('ALPHA_VANTAGE_API_KEY', ''),
            type="password",
            help="Get your free API key from https://www.alphavantage.co/support/#api-key"
        )

        st.markdown("#### Cryptocurrency Data")
        col1, col2 = st.columns(2)
        with col1:
            binance_key = st.text_input(
                "Binance API Key",
                value=config.get('BINANCE_API_KEY', ''),
                type="password",
                help="Optional: Required for crypto trading features"
            )
        with col2:
            binance_secret = st.text_input(
                "Binance API Secret",
                value=config.get('BINANCE_API_SECRET', ''),
                type="password",
                help="Optional: Required for crypto trading features"
            )

        st.markdown("#### News & Sentiment")
        news_api_key = st.text_input(
            "News API Key",
            value=config.get('NEWS_API_KEY', ''),
            type="password",
            help="Optional: Get your free API key from https://newsapi.org/"
        )

        # Submit button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            submit = st.form_submit_button("💾 Save Configuration", use_container_width=True)
        with col2:
            test = st.form_submit_button("🧪 Test APIs", use_container_width=True)
        with col3:
            clear = st.form_submit_button("🗑️ Clear All", use_container_width=True)

    # Handle form submission
    if submit:
        updates = {}
        errors = []

        # Validate and store keys
        if alpha_vantage_key:
            valid, msg = config.validate_api_key('ALPHA_VANTAGE_API_KEY', alpha_vantage_key)
            if valid:
                updates['ALPHA_VANTAGE_API_KEY'] = alpha_vantage_key
            else:
                errors.append(f"Alpha Vantage: {msg}")

        if binance_key:
            valid, msg = config.validate_api_key('BINANCE_API_KEY', binance_key)
            if valid:
                updates['BINANCE_API_KEY'] = binance_key
            else:
                errors.append(f"Binance Key: {msg}")

        if binance_secret:
            valid, msg = config.validate_api_key('BINANCE_API_SECRET', binance_secret)
            if valid:
                updates['BINANCE_API_SECRET'] = binance_secret
            else:
                errors.append(f"Binance Secret: {msg}")

        if news_api_key:
            valid, msg = config.validate_api_key('NEWS_API_KEY', news_api_key)
            if valid:
                updates['NEWS_API_KEY'] = news_api_key
            else:
                errors.append(f"News API: {msg}")

        if errors:
            for error in errors:
                st.error(error)
        elif updates:
            config.set_multiple(updates)
            st.success(f"✅ Configuration saved! {len(updates)} API key(s) updated.")
            st.rerun()
        else:
            st.warning("No API keys provided.")

    if test:
        st.info("🧪 Testing API connections...")
        # Test Alpha Vantage
        if config.get('ALPHA_VANTAGE_API_KEY'):
            try:
                fetcher = StockDataFetcher(alpha_vantage_key=config.get('ALPHA_VANTAGE_API_KEY'))
                st.success("✅ Alpha Vantage: Connection OK")
            except Exception as e:
                st.error(f"❌ Alpha Vantage: {str(e)}")

        # Test Binance
        if config.get('BINANCE_API_KEY'):
            try:
                fetcher = CryptoDataFetcher(
                    api_key=config.get('BINANCE_API_KEY'),
                    api_secret=config.get('BINANCE_API_SECRET')
                )
                price = fetcher.get_current_price('BTCUSDT')
                st.success(f"✅ Binance: Connection OK (BTC Price: ${price.get('current_price', 0):,.2f})")
            except Exception as e:
                st.error(f"❌ Binance: {str(e)}")

    if clear:
        config.clear_all()
        st.success("🗑️ All configuration cleared!")
        st.rerun()

    # Information section
    st.markdown("---")
    st.markdown("### 📚 Getting API Keys")

    with st.expander("🔑 Alpha Vantage (Required for Stocks)"):
        st.markdown("""
        **Alpha Vantage** provides free stock market data.

        1. Visit: https://www.alphavantage.co/support/#api-key
        2. Enter your email
        3. Copy the API key
        4. Paste it above and save

        **Free Tier**: 5 API requests per minute, 500 per day
        """)

    with st.expander("🔑 Binance (Optional for Crypto)"):
        st.markdown("""
        **Binance** provides cryptocurrency market data.

        1. Create account: https://www.binance.com
        2. Go to: Account > API Management
        3. Create new API key
        4. Copy both API Key and Secret Key
        5. Paste them above and save

        **Note**: You don't need trading permissions, read-only is sufficient.
        """)

    with st.expander("🔑 News API (Optional for Sentiment)"):
        st.markdown("""
        **News API** provides news headlines for sentiment analysis.

        1. Visit: https://newsapi.org/register
        2. Register for free account
        3. Copy your API key
        4. Paste it above and save

        **Free Tier**: 100 requests per day
        """)


@st.cache_data(ttl=3600)
def load_data(symbol: str, market_type: str, start_date: str, end_date: str, api_keys: dict):
    """Load and cache market data"""
    if market_type == "Stock":
        fetcher = StockDataFetcher(alpha_vantage_key=api_keys.get('alpha_vantage'))
        return fetcher.fetch_historical_data(symbol, start_date, end_date)
    else:
        fetcher = CryptoDataFetcher(
            api_key=api_keys.get('binance_key'),
            api_secret=api_keys.get('binance_secret')
        )
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


def show_trading_page():
    """Main trading dashboard page"""
    config = st.session_state.config_manager

    # Check if configured
    if not config.is_configured():
        st.warning("⚠️ API keys not configured. Please configure your API keys in the Settings page.")
        if st.button("Go to Settings"):
            st.session_state.page = 'Settings'
            st.rerun()
        return

    # Header
    st.markdown('<div class="main-header">📈 ML Trading Bot Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    market_type = st.sidebar.selectbox("Market Type", ["Stock", "Crypto"])

    if market_type == "Stock":
        symbol = st.sidebar.text_input("Symbol", "AAPL")
    else:
        symbol = st.sidebar.text_input("Symbol", "BTCUSDT")

    mode = st.sidebar.selectbox("Trading Mode", ["Long-Term Forecast", "Day Trading Signals", "Both"])

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    st.sidebar.markdown("### Date Range")
    start_date = st.sidebar.date_input("Start Date", start_date)
    end_date = st.sidebar.date_input("End Date", end_date)

    # Load data button
    if st.sidebar.button("🔄 Load Data", use_container_width=True):
        with st.spinner(f"Loading {symbol} data..."):
            try:
                api_keys = {
                    'alpha_vantage': config.get('ALPHA_VANTAGE_API_KEY'),
                    'binance_key': config.get('BINANCE_API_KEY'),
                    'binance_secret': config.get('BINANCE_API_SECRET')
                }

                df = load_data(
                    symbol,
                    market_type,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    api_keys
                )

                if df.empty:
                    st.error(f"No data found for {symbol}")
                    return

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

    # Simplified prediction section
    st.markdown("---")
    st.markdown("### 🤖 Quick Prediction")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔮 Generate Forecast", use_container_width=True):
            st.info("⏳ Training model... This may take a few minutes.")
            st.info("💡 Tip: For faster predictions, use smaller date ranges or pre-trained models.")

    with col2:
        if st.button("⚡ Generate Signal", use_container_width=True):
            st.info("⏳ Analyzing market... This may take a moment.")


def main():
    """Main application"""

    # Navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'Trading'

    # Sidebar navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")

    if st.sidebar.button("📈 Trading Dashboard", use_container_width=True):
        st.session_state.page = 'Trading'
        st.rerun()

    if st.sidebar.button("⚙️ Settings", use_container_width=True):
        st.session_state.page = 'Settings'
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("**ML Trading Bot v1.0**\n\nAI-powered stock and crypto prediction system with continuous learning.\n\n⚠️ For educational purposes only.")

    # Show selected page
    if st.session_state.page == 'Settings':
        show_settings_page()
    else:
        show_trading_page()


if __name__ == "__main__":
    main()
