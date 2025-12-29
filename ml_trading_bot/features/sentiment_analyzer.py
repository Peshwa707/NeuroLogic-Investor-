"""
Sentiment analysis module for news and social media
"""
import logging
from typing import List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

logger = logging.getLogger(__name__)

# Optional: Use transformers for advanced sentiment analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using VADER only.")


class SentimentAnalyzer:
    """VADER-based sentiment analyzer for financial text"""

    def __init__(self):
        """Initialize VADER sentiment analyzer"""
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER sentiment analyzer initialized")

    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment of a single text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        try:
            scores = self.analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],  # Overall sentiment (-1 to 1)
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    def analyze_headlines(self, headlines: List[str]) -> dict:
        """
        Analyze sentiment from multiple headlines

        Args:
            headlines: List of headlines

        Returns:
            Dictionary with aggregated sentiment scores
        """
        if not headlines:
            return {
                'avg_compound': 0.0,
                'avg_positive': 0.0,
                'avg_negative': 0.0,
                'bullish_ratio': 0.5,
                'bearish_ratio': 0.5,
                'total_articles': 0
            }

        scores = [self.analyze_text(headline) for headline in headlines]

        compounds = [s['compound'] for s in scores]
        positives = [s['positive'] for s in scores]
        negatives = [s['negative'] for s in scores]

        # Count bullish (positive) and bearish (negative) sentiment
        bullish_count = sum(1 for c in compounds if c > 0.05)
        bearish_count = sum(1 for c in compounds if c < -0.05)

        return {
            'avg_compound': np.mean(compounds),
            'avg_positive': np.mean(positives),
            'avg_negative': np.mean(negatives),
            'bullish_ratio': bullish_count / len(headlines) if headlines else 0,
            'bearish_ratio': bearish_count / len(headlines) if headlines else 0,
            'total_articles': len(headlines)
        }

    def get_sentiment_signal(self, compound_score: float) -> str:
        """
        Convert compound score to trading signal

        Args:
            compound_score: Compound sentiment score (-1 to 1)

        Returns:
            Signal: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if compound_score > 0.25:
            return 'BULLISH'
        elif compound_score < -0.25:
            return 'BEARISH'
        else:
            return 'NEUTRAL'


class BERTSentimentAnalyzer:
    """BERT-based sentiment analyzer using FinBERT"""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize BERT sentiment analyzer

        Args:
            model_name: Hugging Face model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")

        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name
            )
            logger.info(f"BERT sentiment analyzer initialized with {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize BERT analyzer: {str(e)}")
            raise

    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment using BERT

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and score
        """
        try:
            # Truncate text to model's max length
            text = text[:512]

            result = self.classifier(text)[0]

            # Convert label to standardized format
            label_map = {
                'positive': 'BULLISH',
                'negative': 'BEARISH',
                'neutral': 'NEUTRAL'
            }

            return {
                'label': label_map.get(result['label'].lower(), result['label']),
                'confidence': result['score'],
                'raw_label': result['label']
            }

        except Exception as e:
            logger.error(f"Error analyzing text with BERT: {str(e)}")
            return {'label': 'NEUTRAL', 'confidence': 0.0, 'raw_label': 'unknown'}

    def analyze_headlines(self, headlines: List[str]) -> dict:
        """
        Analyze sentiment from multiple headlines using BERT

        Args:
            headlines: List of headlines

        Returns:
            Dictionary with aggregated sentiment
        """
        if not headlines:
            return {
                'avg_confidence': 0.0,
                'bullish_ratio': 0.5,
                'bearish_ratio': 0.5,
                'total_articles': 0
            }

        results = [self.analyze_text(headline) for headline in headlines]

        bullish_count = sum(1 for r in results if r['label'] == 'BULLISH')
        bearish_count = sum(1 for r in results if r['label'] == 'BEARISH')
        confidences = [r['confidence'] for r in results]

        return {
            'avg_confidence': np.mean(confidences),
            'bullish_ratio': bullish_count / len(headlines) if headlines else 0,
            'bearish_ratio': bearish_count / len(headlines) if headlines else 0,
            'neutral_ratio': 1 - (bullish_count + bearish_count) / len(headlines) if headlines else 0,
            'total_articles': len(headlines)
        }


class SentimentFeatureEngineer:
    """Engineer sentiment features for ML models"""

    def __init__(self, analyzer_type: str = 'vader'):
        """
        Initialize sentiment feature engineer

        Args:
            analyzer_type: 'vader' or 'bert'
        """
        if analyzer_type == 'vader':
            self.analyzer = SentimentAnalyzer()
        elif analyzer_type == 'bert' and TRANSFORMERS_AVAILABLE:
            self.analyzer = BERTSentimentAnalyzer()
        else:
            logger.warning(f"Analyzer type '{analyzer_type}' not available, using VADER")
            self.analyzer = SentimentAnalyzer()

        self.analyzer_type = analyzer_type

    def add_sentiment_features(self, headlines: List[str]) -> dict:
        """
        Generate sentiment features from headlines

        Args:
            headlines: List of news headlines

        Returns:
            Dictionary of sentiment features
        """
        sentiment_data = self.analyzer.analyze_headlines(headlines)

        if self.analyzer_type == 'vader':
            features = {
                'sentiment_compound': sentiment_data['avg_compound'],
                'sentiment_positive': sentiment_data['avg_positive'],
                'sentiment_negative': sentiment_data['avg_negative'],
                'sentiment_bullish_ratio': sentiment_data['bullish_ratio'],
                'sentiment_bearish_ratio': sentiment_data['bearish_ratio']
            }
        else:  # BERT
            features = {
                'sentiment_confidence': sentiment_data['avg_confidence'],
                'sentiment_bullish_ratio': sentiment_data['bullish_ratio'],
                'sentiment_bearish_ratio': sentiment_data['bearish_ratio'],
                'sentiment_neutral_ratio': sentiment_data['neutral_ratio']
            }

        return features


# Mock news fetcher (in production, integrate with News API, Twitter API, etc.)
class MockNewsFetcher:
    """Mock news fetcher for demonstration"""

    def fetch_news(self, symbol: str, days: int = 7) -> List[str]:
        """
        Fetch news headlines for a symbol (mock implementation)

        Args:
            symbol: Stock/crypto symbol
            days: Number of days to look back

        Returns:
            List of headlines
        """
        # In production, integrate with:
        # - News API (newsapi.org)
        # - Alpha Vantage news endpoint
        # - Twitter API
        # - Reddit API
        # - Financial news RSS feeds

        logger.info(f"Fetching news for {symbol} (mock)")

        # Return empty list as placeholder
        return []


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG

    logging.config.dictConfig(LOGGING_CONFIG)

    # Sample headlines
    headlines = [
        "Company reports record profits, stock surges",
        "Market analysts predict strong growth for tech sector",
        "Economic indicators show signs of recession",
        "Company faces regulatory investigation",
        "New product launch exceeds expectations"
    ]

    # VADER analysis
    print("=== VADER Sentiment Analysis ===")
    vader_analyzer = SentimentAnalyzer()

    for headline in headlines:
        sentiment = vader_analyzer.analyze_text(headline)
        signal = vader_analyzer.get_sentiment_signal(sentiment['compound'])
        print(f"\n{headline}")
        print(f"Sentiment: {sentiment['compound']:.3f} ({signal})")

    # Aggregate sentiment
    aggregate = vader_analyzer.analyze_headlines(headlines)
    print(f"\n\nAggregate Sentiment:")
    print(f"Average Compound: {aggregate['avg_compound']:.3f}")
    print(f"Bullish Ratio: {aggregate['bullish_ratio']:.2%}")
    print(f"Bearish Ratio: {aggregate['bearish_ratio']:.2%}")

    # Feature engineering
    feature_engineer = SentimentFeatureEngineer('vader')
    features = feature_engineer.add_sentiment_features(headlines)
    print(f"\n\nSentiment Features:")
    for key, value in features.items():
        print(f"{key}: {value:.4f}")
