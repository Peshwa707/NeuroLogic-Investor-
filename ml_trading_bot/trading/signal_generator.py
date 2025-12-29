"""
Day trading signal generator
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DayTradingSignalGenerator:
    """Generate real-time trading signals for day trading"""

    def __init__(self, classifier, feature_engineer):
        """
        Initialize signal generator

        Args:
            classifier: Trained DayTradingClassifier instance
            feature_engineer: FeatureEngineer instance
        """
        self.classifier = classifier
        self.feature_engineer = feature_engineer

        logger.info("DayTradingSignalGenerator initialized")

    def generate_signal(self, current_data: pd.DataFrame,
                       buy_threshold: float = 0.6,
                       sell_threshold: float = 0.4) -> Dict:
        """
        Generate real-time trading signal

        Args:
            current_data: Current market data
            buy_threshold: Probability threshold for BUY signal
            sell_threshold: Probability threshold for SELL signal

        Returns:
            Dictionary with signal and metadata
        """
        try:
            logger.info("Generating trading signal...")

            # Engineer features
            self.feature_engineer.df = current_data.copy()
            features = (self.feature_engineer
                       .add_moving_averages()
                       .add_macd()
                       .add_bollinger_bands()
                       .add_rsi()
                       .add_stochastic_oscillator()
                       .add_atr()
                       .add_price_features()
                       .add_volume_features()
                       .get_features())

            if features.empty:
                logger.error("No features generated")
                return self._empty_signal()

            # Get latest features
            latest_features = features.iloc[-1:]

            # Get prediction
            signal = self.classifier.predict_signal(
                latest_features,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold
            )

            # Get confidence
            confidence = self.classifier.predict_proba(latest_features)[0]

            # Get current price
            current_price = current_data['close'].iloc[-1]

            # Additional signal metadata
            metadata = self._generate_metadata(features, signal, confidence)

            result = {
                'signal': signal,
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat(),
                'price': float(current_price),
                'symbol': current_data['symbol'].iloc[-1] if 'symbol' in current_data.columns else 'UNKNOWN',
                'metadata': metadata
            }

            logger.info(
                f"Signal generated: {signal} (confidence: {confidence:.2%}) "
                f"at price ${current_price:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return self._empty_signal()

    def _generate_metadata(self, features: pd.DataFrame,
                          signal: str, confidence: float) -> Dict:
        """
        Generate additional signal metadata

        Args:
            features: Feature DataFrame
            signal: Generated signal
            confidence: Signal confidence

        Returns:
            Dictionary with metadata
        """
        try:
            latest = features.iloc[-1]

            metadata = {
                'technical_indicators': {
                    'RSI': float(latest.get('RSI', 0)),
                    'MACD': float(latest.get('MACD', 0)),
                    'MACD_signal': float(latest.get('MACD_signal', 0)),
                    'BB_pct': float(latest.get('BB_pct', 0.5)),
                    'Stoch_K': float(latest.get('Stoch_K', 50)),
                    'ATR_pct': float(latest.get('ATR_pct', 0))
                },
                'price_metrics': {
                    'SMA_20': float(latest.get('SMA_20', 0)),
                    'SMA_50': float(latest.get('SMA_50', 0)),
                    'price_change_pct': float(latest.get('price_change_pct', 0)),
                    'volume_ratio': float(latest.get('volume_ratio', 1.0))
                },
                'signal_strength': self._calculate_signal_strength(latest, signal, confidence)
            }

            return metadata

        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            return {}

    def _calculate_signal_strength(self, features: pd.Series,
                                   signal: str, confidence: float) -> str:
        """
        Calculate signal strength based on technical indicators

        Args:
            features: Feature series
            signal: Trading signal
            confidence: Signal confidence

        Returns:
            Signal strength ('STRONG', 'MODERATE', 'WEAK')
        """
        try:
            strength_score = confidence  # Start with model confidence

            # Add indicator confirmations
            rsi = features.get('RSI', 50)
            macd = features.get('MACD', 0)
            macd_signal = features.get('MACD_signal', 0)

            if signal == 'BUY':
                # Bullish confirmations
                if rsi < 40:  # Oversold
                    strength_score += 0.15
                if macd > macd_signal:  # MACD crossover
                    strength_score += 0.15

            elif signal == 'SELL':
                # Bearish confirmations
                if rsi > 60:  # Overbought
                    strength_score += 0.15
                if macd < macd_signal:  # MACD crossover
                    strength_score += 0.15

            # Classify strength
            if strength_score > 0.75:
                return 'STRONG'
            elif strength_score > 0.55:
                return 'MODERATE'
            else:
                return 'WEAK'

        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return 'UNKNOWN'

    def _empty_signal(self) -> Dict:
        """Return empty signal dict"""
        return {
            'signal': 'HOLD',
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'price': 0.0,
            'symbol': 'UNKNOWN',
            'metadata': {},
            'error': 'signal_generation_failed'
        }

    def batch_generate_signals(self, data_dict: Dict[str, pd.DataFrame],
                               buy_threshold: float = 0.6,
                               sell_threshold: float = 0.4) -> Dict[str, Dict]:
        """
        Generate signals for multiple symbols

        Args:
            data_dict: Dictionary mapping symbols to their data
            buy_threshold: Buy signal threshold
            sell_threshold: Sell signal threshold

        Returns:
            Dictionary mapping symbols to their signals
        """
        signals = {}

        for symbol, data in data_dict.items():
            logger.info(f"Generating signal for {symbol}...")
            signals[symbol] = self.generate_signal(
                data,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold
            )

        logger.info(f"Generated signals for {len(signals)} symbols")
        return signals


class SignalValidator:
    """Validate trading signals before execution"""

    @staticmethod
    def validate_signal(signal: Dict, min_confidence: float = 0.5,
                       max_volatility: float = 5.0) -> bool:
        """
        Validate if signal meets execution criteria

        Args:
            signal: Signal dictionary
            min_confidence: Minimum confidence threshold
            max_volatility: Maximum ATR percentage threshold

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Check confidence
            if signal['confidence'] < min_confidence:
                logger.warning(
                    f"Signal rejected: confidence {signal['confidence']:.2%} "
                    f"< threshold {min_confidence:.2%}"
                )
                return False

            # Check volatility (if available)
            metadata = signal.get('metadata', {})
            technical = metadata.get('technical_indicators', {})
            atr_pct = technical.get('ATR_pct', 0)

            if atr_pct > max_volatility:
                logger.warning(
                    f"Signal rejected: volatility {atr_pct:.2f}% "
                    f"> threshold {max_volatility:.2f}%"
                )
                return False

            # Check signal is not HOLD
            if signal['signal'] == 'HOLD':
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating signal: {str(e)}")
            return False

    @staticmethod
    def apply_risk_management(signal: Dict, current_position: Optional[str] = None,
                             max_position_size: float = 1.0) -> Dict:
        """
        Apply risk management rules to signal

        Args:
            signal: Signal dictionary
            current_position: Current position ('LONG', 'SHORT', None)
            max_position_size: Maximum position size (1.0 = 100%)

        Returns:
            Modified signal with position sizing
        """
        try:
            # Don't open opposite positions
            if current_position == 'LONG' and signal['signal'] == 'SELL':
                signal['action'] = 'CLOSE_LONG'
            elif current_position == 'SHORT' and signal['signal'] == 'BUY':
                signal['action'] = 'CLOSE_SHORT'
            else:
                signal['action'] = signal['signal']

            # Position sizing based on confidence
            confidence = signal['confidence']
            position_size = min(confidence * max_position_size, max_position_size)

            signal['position_size'] = float(position_size)

            return signal

        except Exception as e:
            logger.error(f"Error applying risk management: {str(e)}")
            return signal


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG
    from ml_trading_bot.data.ingestion import StockDataFetcher
    from ml_trading_bot.features import FeatureEngineer
    from ml_trading_bot.models import DayTradingClassifier

    logging.config.dictConfig(LOGGING_CONFIG)

    print("=== Day Trading Signal Generator Example ===\n")

    # Fetch data
    print("Fetching data...")
    fetcher = StockDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', '2023-01-01', '2024-01-01')

    # Engineer features and train classifier
    print("Training classifier...")
    engineer = FeatureEngineer(df)
    features_df = engineer.add_all_indicators().get_features()

    classifier = DayTradingClassifier(threshold=0.005)
    features_df['target'] = classifier.create_target(features_df)
    X, y = classifier.prepare_features(features_df)
    classifier.train(X, y)

    # Create signal generator
    print("\nCreating signal generator...")
    signal_gen = DayTradingSignalGenerator(classifier, FeatureEngineer(df))

    # Generate signal
    print("\nGenerating trading signal...")
    signal = signal_gen.generate_signal(df.tail(100))

    print(f"\nSignal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Price: ${signal['price']:.2f}")
    print(f"Signal Strength: {signal['metadata']['signal_strength']}")
    print(f"\nTechnical Indicators:")
    for indicator, value in signal['metadata']['technical_indicators'].items():
        print(f"  {indicator}: {value:.4f}")

    # Validate signal
    print("\nValidating signal...")
    validator = SignalValidator()
    is_valid = validator.validate_signal(signal, min_confidence=0.5)
    print(f"Signal valid: {is_valid}")

    if is_valid:
        risk_managed_signal = validator.apply_risk_management(signal)
        print(f"Action: {risk_managed_signal.get('action', 'N/A')}")
        print(f"Position Size: {risk_managed_signal.get('position_size', 0):.2%}")
