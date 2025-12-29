"""
TimescaleDB storage layer for time-series market data
"""
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from typing import Optional, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TimeSeriesDB:
    """TimescaleDB interface for storing and querying market data"""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """
        Initialize TimescaleDB connection

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        self.connection_string = (
            f"host={host} port={port} dbname={database} "
            f"user={user} password={password}"
        )
        self.conn = None
        self.connect()

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.conn.autocommit = False
            logger.info("Successfully connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {str(e)}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from TimescaleDB")

    def create_tables(self):
        """Create necessary tables and hypertables"""
        try:
            cursor = self.conn.cursor()

            # Create stock_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    UNIQUE(symbol, timestamp)
                );
            """)

            # Convert to hypertable
            cursor.execute("""
                SELECT create_hypertable('stock_data', 'timestamp',
                    if_not_exists => TRUE);
            """)

            # Create crypto_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crypto_data (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    UNIQUE(symbol, timestamp)
                );
            """)

            # Convert to hypertable
            cursor.execute("""
                SELECT create_hypertable('crypto_data', 'timestamp',
                    if_not_exists => TRUE);
            """)

            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    predicted_price DOUBLE PRECISION,
                    actual_price DOUBLE PRECISION,
                    confidence DOUBLE PRECISION,
                    model_version TEXT,
                    error DOUBLE PRECISION
                );
            """)

            cursor.execute("""
                SELECT create_hypertable('predictions', 'timestamp',
                    if_not_exists => TRUE);
            """)

            # Create trading_signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence DOUBLE PRECISION,
                    current_price DOUBLE PRECISION,
                    metadata JSONB
                );
            """)

            cursor.execute("""
                SELECT create_hypertable('trading_signals', 'timestamp',
                    if_not_exists => TRUE);
            """)

            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_symbol
                ON stock_data (symbol, timestamp DESC);
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_crypto_symbol
                ON crypto_data (symbol, timestamp DESC);
            """)

            self.conn.commit()
            logger.info("Successfully created tables and hypertables")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating tables: {str(e)}")
            raise

    def store_ohlcv(self, df: pd.DataFrame, table: str = 'stock_data'):
        """
        Store OHLCV data in TimescaleDB

        Args:
            df: DataFrame with columns: timestamp, symbol, open, high, low, close, volume
            table: Table name ('stock_data' or 'crypto_data')
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to store")
            return

        try:
            cursor = self.conn.cursor()

            # Prepare data for insertion
            records = []
            for _, row in df.iterrows():
                records.append((
                    row['timestamp'],
                    row['symbol'],
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                ))

            # Use execute_values for efficient batch insertion
            execute_values(
                cursor,
                f"""
                INSERT INTO {table} (timestamp, symbol, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                records
            )

            self.conn.commit()
            logger.info(f"Successfully stored {len(records)} records in {table}")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing OHLCV data: {str(e)}")
            raise

    def query_data(self, symbol: str, start: Optional[datetime] = None,
                   end: Optional[datetime] = None, table: str = 'stock_data') -> pd.DataFrame:
        """
        Query OHLCV data for a symbol

        Args:
            symbol: Symbol to query
            start: Start timestamp (optional)
            end: End timestamp (optional)
            table: Table name ('stock_data' or 'crypto_data')

        Returns:
            DataFrame with OHLCV data
        """
        try:
            query = f"SELECT * FROM {table} WHERE symbol = %s"
            params = [symbol]

            if start:
                query += " AND timestamp >= %s"
                params.append(start)

            if end:
                query += " AND timestamp <= %s"
                params.append(end)

            query += " ORDER BY timestamp ASC"

            df = pd.read_sql_query(query, self.conn, params=params)
            logger.info(f"Retrieved {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error querying data: {str(e)}")
            return pd.DataFrame()

    def query_sliding_window(self, symbol: str, days: int,
                            table: str = 'stock_data') -> pd.DataFrame:
        """
        Query data for a sliding window (last N days)

        Args:
            symbol: Symbol to query
            days: Number of days to look back
            table: Table name

        Returns:
            DataFrame with data from last N days
        """
        query = f"""
            SELECT * FROM {table}
            WHERE symbol = %s
            AND timestamp >= NOW() - INTERVAL '{days} days'
            ORDER BY timestamp ASC
        """

        try:
            df = pd.read_sql_query(query, self.conn, params=[symbol])
            logger.info(f"Retrieved {len(df)} records for {symbol} (last {days} days)")
            return df

        except Exception as e:
            logger.error(f"Error querying sliding window: {str(e)}")
            return pd.DataFrame()

    def store_prediction(self, symbol: str, prediction_type: str,
                        predicted_price: float, actual_price: Optional[float] = None,
                        confidence: Optional[float] = None, model_version: Optional[str] = None):
        """
        Store a prediction in the database

        Args:
            symbol: Symbol
            prediction_type: Type of prediction ('long_term', 'day_trading')
            predicted_price: Predicted price
            actual_price: Actual price (if known)
            confidence: Model confidence
            model_version: Model version string
        """
        try:
            cursor = self.conn.cursor()

            error = None
            if actual_price is not None and predicted_price is not None:
                error = abs(predicted_price - actual_price) / actual_price if actual_price != 0 else 0

            cursor.execute("""
                INSERT INTO predictions
                (timestamp, symbol, prediction_type, predicted_price, actual_price,
                 confidence, model_version, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (datetime.now(), symbol, prediction_type, predicted_price,
                  actual_price, confidence, model_version, error))

            self.conn.commit()
            logger.info(f"Stored prediction for {symbol}: ${predicted_price:.2f}")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing prediction: {str(e)}")

    def store_trading_signal(self, symbol: str, signal: str, confidence: float,
                            current_price: float, metadata: Optional[dict] = None):
        """
        Store a trading signal

        Args:
            symbol: Symbol
            signal: Signal type ('BUY', 'SELL', 'HOLD')
            confidence: Signal confidence
            current_price: Current price
            metadata: Additional metadata (JSON)
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                INSERT INTO trading_signals
                (timestamp, symbol, signal, confidence, current_price, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (datetime.now(), symbol, signal, confidence, current_price,
                  psycopg2.extras.Json(metadata) if metadata else None))

            self.conn.commit()
            logger.info(f"Stored trading signal for {symbol}: {signal} (confidence: {confidence:.2f})")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing trading signal: {str(e)}")

    def get_prediction_metrics(self, symbol: str, days: int = 30) -> dict:
        """
        Calculate prediction performance metrics

        Args:
            symbol: Symbol to analyze
            days: Number of days to look back

        Returns:
            Dictionary with performance metrics
        """
        try:
            query = """
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(error) as avg_error,
                    STDDEV(error) as std_error,
                    MIN(error) as min_error,
                    MAX(error) as max_error
                FROM predictions
                WHERE symbol = %s
                AND actual_price IS NOT NULL
                AND timestamp >= NOW() - INTERVAL '%s days'
            """

            cursor = self.conn.cursor()
            cursor.execute(query, (symbol, days))
            result = cursor.fetchone()

            metrics = {
                'total_predictions': result[0] or 0,
                'avg_error': result[1] or 0,
                'std_error': result[2] or 0,
                'min_error': result[3] or 0,
                'max_error': result[4] or 0
            }

            logger.info(f"Calculated prediction metrics for {symbol}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating prediction metrics: {str(e)}")
            return {}

    def get_all_symbols(self, table: str = 'stock_data') -> List[str]:
        """
        Get list of all symbols in the database

        Args:
            table: Table name

        Returns:
            List of unique symbols
        """
        try:
            query = f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol"
            cursor = self.conn.cursor()
            cursor.execute(query)
            symbols = [row[0] for row in cursor.fetchall()]

            logger.info(f"Found {len(symbols)} symbols in {table}")
            return symbols

        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            return []


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG, TIMESCALE_CONFIG

    logging.config.dictConfig(LOGGING_CONFIG)

    # Initialize database
    db = TimeSeriesDB(**TIMESCALE_CONFIG)

    # Create tables
    db.create_tables()

    # Example: Store sample data
    sample_data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'symbol': ['AAPL'],
        'open': [150.0],
        'high': [155.0],
        'low': [149.0],
        'close': [154.0],
        'volume': [1000000]
    })

    db.store_ohlcv(sample_data)

    # Query data
    df = db.query_sliding_window('AAPL', days=30)
    print(f"Retrieved {len(df)} records")

    db.disconnect()
