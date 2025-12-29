"""
Kafka message queue manager for event-driven architecture
"""
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import json
import logging
from typing import Callable, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class KafkaManager:
    """Kafka producer and consumer manager"""

    def __init__(self, bootstrap_servers: str):
        """
        Initialize Kafka manager

        Args:
            bootstrap_servers: Kafka bootstrap servers (e.g., 'localhost:9092')
        """
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None

    def create_producer(self) -> KafkaProducer:
        """
        Create Kafka producer

        Returns:
            KafkaProducer instance
        """
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info("Kafka producer created successfully")
            return self.producer

        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {str(e)}")
            raise

    def create_consumer(self, topics: list, group_id: str,
                       auto_offset_reset: str = 'earliest') -> KafkaConsumer:
        """
        Create Kafka consumer

        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID
            auto_offset_reset: Where to start reading ('earliest' or 'latest')

        Returns:
            KafkaConsumer instance
        """
        try:
            self.consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset=auto_offset_reset,
                enable_auto_commit=True
            )
            logger.info(f"Kafka consumer created for topics: {topics}")
            return self.consumer

        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {str(e)}")
            raise

    def send_message(self, topic: str, value: dict, key: Optional[str] = None) -> bool:
        """
        Send message to Kafka topic

        Args:
            topic: Topic name
            value: Message value (dict)
            key: Message key (optional)

        Returns:
            True if successful, False otherwise
        """
        if not self.producer:
            self.create_producer()

        try:
            # Add timestamp if not present
            if 'timestamp' not in value:
                value['timestamp'] = datetime.now().isoformat()

            future = self.producer.send(topic, value=value, key=key)

            # Wait for confirmation
            record_metadata = future.get(timeout=10)

            logger.info(
                f"Message sent to {topic} "
                f"(partition: {record_metadata.partition}, offset: {record_metadata.offset})"
            )
            return True

        except KafkaError as e:
            logger.error(f"Failed to send message to {topic}: {str(e)}")
            return False

    def send_stock_data(self, data: dict):
        """
        Send stock data to Kafka

        Args:
            data: Stock data dictionary
        """
        return self.send_message('stock-prices', data, key=data.get('symbol'))

    def send_crypto_data(self, data: dict):
        """
        Send crypto data to Kafka

        Args:
            data: Crypto data dictionary
        """
        return self.send_message('crypto-prices', data, key=data.get('symbol'))

    def send_prediction(self, prediction: dict):
        """
        Send prediction to Kafka

        Args:
            prediction: Prediction dictionary
        """
        return self.send_message('predictions', prediction, key=prediction.get('symbol'))

    def send_trading_signal(self, signal: dict):
        """
        Send trading signal to Kafka

        Args:
            signal: Trading signal dictionary
        """
        return self.send_message('trading-signals', signal, key=signal.get('symbol'))

    def consume_messages(self, callback: Callable[[Any], None], max_messages: Optional[int] = None):
        """
        Consume messages from Kafka topics

        Args:
            callback: Function to handle each message
            max_messages: Maximum number of messages to consume (None for infinite)
        """
        if not self.consumer:
            logger.error("Consumer not initialized")
            return

        try:
            message_count = 0
            logger.info("Started consuming messages...")

            for message in self.consumer:
                try:
                    # Call callback with message value
                    callback(message.value)

                    message_count += 1
                    if max_messages and message_count >= max_messages:
                        logger.info(f"Consumed {message_count} messages, stopping")
                        break

                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    continue

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Error consuming messages: {str(e)}")
        finally:
            self.close()

    def close(self):
        """Close producer and consumer"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")

        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")


class DataIngestionPipeline:
    """Pipeline for ingesting data and publishing to Kafka"""

    def __init__(self, kafka_manager: KafkaManager):
        """
        Initialize data ingestion pipeline

        Args:
            kafka_manager: KafkaManager instance
        """
        self.kafka = kafka_manager

    def ingest_stock_data(self, fetcher, symbol: str, start: str, end: str):
        """
        Fetch stock data and publish to Kafka

        Args:
            fetcher: StockDataFetcher instance
            symbol: Stock symbol
            start: Start date
            end: End date
        """
        logger.info(f"Ingesting stock data for {symbol}")

        df = fetcher.fetch_historical_data(symbol, start, end)

        if df.empty:
            logger.warning(f"No data to ingest for {symbol}")
            return

        # Publish each row to Kafka
        for _, row in df.iterrows():
            data = {
                'symbol': row['symbol'],
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            }
            self.kafka.send_stock_data(data)

        logger.info(f"Ingested {len(df)} records for {symbol}")

    def ingest_crypto_data(self, fetcher, symbol: str, interval: str, start: str):
        """
        Fetch crypto data and publish to Kafka

        Args:
            fetcher: CryptoDataFetcher instance
            symbol: Crypto symbol
            interval: Time interval
            start: Start date
        """
        logger.info(f"Ingesting crypto data for {symbol}")

        df = fetcher.get_historical_klines(symbol, interval, start)

        if df.empty:
            logger.warning(f"No data to ingest for {symbol}")
            return

        # Publish each row to Kafka
        for _, row in df.iterrows():
            data = {
                'symbol': row['symbol'],
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            self.kafka.send_crypto_data(data)

        logger.info(f"Ingested {len(df)} records for {symbol}")


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG, KAFKA_CONFIG

    logging.config.dictConfig(LOGGING_CONFIG)

    # Initialize Kafka manager
    kafka = KafkaManager(KAFKA_CONFIG['bootstrap_servers'])

    # Example: Send a test message
    test_data = {
        'symbol': 'AAPL',
        'timestamp': datetime.now().isoformat(),
        'open': 150.0,
        'high': 155.0,
        'low': 149.0,
        'close': 154.0,
        'volume': 1000000
    }

    kafka.send_stock_data(test_data)

    # Example: Create consumer
    def message_handler(message):
        print(f"Received message: {message}")

    consumer = kafka.create_consumer(['stock-prices'], group_id='test-group')

    # Consume a few messages for testing
    # kafka.consume_messages(message_handler, max_messages=5)

    kafka.close()
