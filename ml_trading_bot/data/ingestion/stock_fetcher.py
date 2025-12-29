"""
Stock data fetcher using Yahoo Finance and Alpha Vantage
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging
from alpha_vantage.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetch stock market data from various sources"""

    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize stock data fetcher

        Args:
            alpha_vantage_key: API key for Alpha Vantage (optional)
        """
        self.alpha_vantage_key = alpha_vantage_key
        if alpha_vantage_key:
            self.ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
        else:
            self.ts = None
            logger.warning("Alpha Vantage API key not provided. Intraday data will be unavailable.")

    def fetch_historical_data(self, symbol: str, start: str, end: str,
                             interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Yahoo Finance

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching historical data for {symbol} from {start} to {end}")
            df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]

            # Add symbol column
            df['symbol'] = symbol

            # Reset index to make timestamp a column
            df.reset_index(inplace=True)
            df.rename(columns={'date': 'timestamp'}, inplace=True)

            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_realtime_quote(self, symbol: str) -> dict:
        """
        Get current quote for a symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with current price data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            quote = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'open': info.get('regularMarketOpen', 0),
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0)
            }

            logger.info(f"Fetched real-time quote for {symbol}: ${quote['current_price']}")
            return quote

        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {str(e)}")
            return {}

    def fetch_intraday_data(self, symbol: str, interval: str = '5min',
                           outputsize: str = 'compact') -> pd.DataFrame:
        """
        Fetch intraday data from Alpha Vantage

        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            outputsize: 'compact' (last 100 data points) or 'full' (full historical)

        Returns:
            DataFrame with intraday data
        """
        if not self.ts:
            logger.error("Alpha Vantage API key not configured")
            return pd.DataFrame()

        try:
            logger.info(f"Fetching intraday data for {symbol} with interval {interval}")
            data, meta = self.ts.get_intraday(symbol=symbol, interval=interval,
                                             outputsize=outputsize)

            # Standardize column names
            data.columns = ['open', 'high', 'low', 'close', 'volume']

            # Add symbol column
            data['symbol'] = symbol

            # Reset index to make timestamp a column
            data.reset_index(inplace=True)
            data.rename(columns={'date': 'timestamp'}, inplace=True)

            logger.info(f"Successfully fetched {len(data)} intraday records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_multiple_symbols(self, symbols: list, start: str, end: str) -> dict:
        """
        Fetch historical data for multiple symbols

        Args:
            symbols: List of stock ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        data_dict = {}

        for symbol in symbols:
            df = self.fetch_historical_data(symbol, start, end)
            if not df.empty:
                data_dict[symbol] = df

        logger.info(f"Fetched data for {len(data_dict)}/{len(symbols)} symbols")
        return data_dict

    def get_latest_n_days(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        Fetch data for the last N days

        Args:
            symbol: Stock ticker symbol
            days: Number of days to fetch

        Returns:
            DataFrame with recent data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Add buffer for weekends

        return self.fetch_historical_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG

    logging.config.dictConfig(LOGGING_CONFIG)

    fetcher = StockDataFetcher()

    # Fetch historical data
    df = fetcher.fetch_historical_data('AAPL', '2023-01-01', '2024-01-01')
    print(f"\nHistorical data shape: {df.shape}")
    print(df.head())

    # Fetch real-time quote
    quote = fetcher.fetch_realtime_quote('AAPL')
    print(f"\nReal-time quote: {quote}")

    # Fetch recent data
    recent_df = fetcher.get_latest_n_days('AAPL', days=30)
    print(f"\nRecent data shape: {recent_df.shape}")
