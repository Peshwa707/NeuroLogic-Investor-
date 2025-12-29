"""
Cryptocurrency data fetcher using Binance API
"""
import pandas as pd
from binance.client import Client
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
from typing import Optional, Callable
import logging
import time

logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """Fetch cryptocurrency data from Binance"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize crypto data fetcher

        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
        """
        self.api_key = api_key
        self.api_secret = api_secret

        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
            logger.info("Binance client initialized with credentials")
        else:
            self.client = Client("", "")  # Public client for market data
            logger.warning("Binance client initialized without credentials (public data only)")

        self.websocket_manager = None

    def get_historical_klines(self, symbol: str, interval: str,
                             start: str, end: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical candlestick data

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Kline interval (1m, 5m, 15m, 1h, 1d, etc.)
            start: Start date (YYYY-MM-DD or timestamp)
            end: End date (YYYY-MM-DD or timestamp), defaults to now

        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching historical klines for {symbol} from {start}")

            if end is None:
                end = datetime.now().strftime('%Y-%m-%d')

            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start,
                end_str=end
            )

            if not klines:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

            # Convert price columns to float
            price_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in price_cols:
                df[col] = df[col].astype(float)

            # Add symbol column
            df['symbol'] = symbol

            # Select relevant columns
            df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical klines for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> dict:
        """
        Get current price for a trading pair

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            Dictionary with current price data
        """
        try:
            ticker = self.client.get_ticker(symbol=symbol)

            price_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': float(ticker['lastPrice']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
                'volume_24h': float(ticker['volume']),
                'price_change_24h': float(ticker['priceChange']),
                'price_change_pct_24h': float(ticker['priceChangePercent'])
            }

            logger.info(f"Fetched current price for {symbol}: ${price_data['current_price']}")
            return price_data

        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            return {}

    def start_websocket_stream(self, symbol: str, callback: Callable,
                               stream_type: str = 'kline') -> ThreadedWebsocketManager:
        """
        Start real-time WebSocket stream

        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            callback: Function to handle incoming messages
            stream_type: Type of stream ('kline', 'ticker', 'trade')

        Returns:
            ThreadedWebsocketManager instance
        """
        try:
            self.websocket_manager = ThreadedWebsocketManager()
            self.websocket_manager.start()

            if stream_type == 'kline':
                self.websocket_manager.start_kline_socket(
                    callback=callback,
                    symbol=symbol.lower(),
                    interval='1m'
                )
                logger.info(f"Started kline WebSocket stream for {symbol}")

            elif stream_type == 'ticker':
                self.websocket_manager.start_symbol_ticker_socket(
                    callback=callback,
                    symbol=symbol.lower()
                )
                logger.info(f"Started ticker WebSocket stream for {symbol}")

            elif stream_type == 'trade':
                self.websocket_manager.start_trade_socket(
                    callback=callback,
                    symbol=symbol.lower()
                )
                logger.info(f"Started trade WebSocket stream for {symbol}")

            return self.websocket_manager

        except Exception as e:
            logger.error(f"Error starting WebSocket stream for {symbol}: {str(e)}")
            return None

    def stop_websocket_stream(self):
        """Stop the WebSocket stream"""
        if self.websocket_manager:
            self.websocket_manager.stop()
            logger.info("WebSocket stream stopped")

    def get_latest_n_days(self, symbol: str, days: int = 60,
                         interval: str = '1d') -> pd.DataFrame:
        """
        Fetch data for the last N days

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            days: Number of days to fetch
            interval: Kline interval

        Returns:
            DataFrame with recent data
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.get_historical_klines(symbol, interval, start_date)

    def get_top_symbols_by_volume(self, quote_asset: str = 'USDT',
                                  limit: int = 10) -> list:
        """
        Get top trading pairs by 24h volume

        Args:
            quote_asset: Quote asset (e.g., 'USDT', 'BTC')
            limit: Number of symbols to return

        Returns:
            List of top symbols
        """
        try:
            tickers = self.client.get_ticker()

            # Filter by quote asset and sort by volume
            filtered = [
                ticker for ticker in tickers
                if ticker['symbol'].endswith(quote_asset)
            ]

            sorted_tickers = sorted(
                filtered,
                key=lambda x: float(x['quoteVolume']),
                reverse=True
            )

            top_symbols = [ticker['symbol'] for ticker in sorted_tickers[:limit]]

            logger.info(f"Top {limit} symbols by volume: {top_symbols}")
            return top_symbols

        except Exception as e:
            logger.error(f"Error fetching top symbols: {str(e)}")
            return []


def websocket_message_handler(msg):
    """Example callback for WebSocket messages"""
    if msg['e'] == 'kline':
        kline = msg['k']
        print(f"Symbol: {kline['s']}, Close: {kline['c']}, Volume: {kline['v']}")


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG

    logging.config.dictConfig(LOGGING_CONFIG)

    fetcher = CryptoDataFetcher()

    # Fetch historical data
    df = fetcher.get_historical_klines('BTCUSDT', '1d', '2023-01-01', '2024-01-01')
    print(f"\nHistorical data shape: {df.shape}")
    print(df.head())

    # Fetch current price
    price = fetcher.get_current_price('BTCUSDT')
    print(f"\nCurrent price: {price}")

    # Get recent data
    recent_df = fetcher.get_latest_n_days('BTCUSDT', days=30)
    print(f"\nRecent data shape: {recent_df.shape}")

    # Get top symbols
    top_symbols = fetcher.get_top_symbols_by_volume(limit=5)
    print(f"\nTop 5 symbols: {top_symbols}")

    # Example WebSocket (uncomment to test)
    # print("\nStarting WebSocket stream for 10 seconds...")
    # fetcher.start_websocket_stream('BTCUSDT', websocket_message_handler)
    # time.sleep(10)
    # fetcher.stop_websocket_stream()
