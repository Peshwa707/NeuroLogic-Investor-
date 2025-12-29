"""
Classification models for day trading signals
"""
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import logging
import pickle
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DayTradingClassifier:
    """XGBoost classifier for day trading signals"""

    def __init__(self, threshold: float = 0.005, n_estimators: int = 100,
                 max_depth: int = 5, learning_rate: float = 0.1):
        """
        Initialize day trading classifier

        Args:
            threshold: Price change threshold for binary classification
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        self.threshold = threshold
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42
        )
        self.feature_names = None
        self.is_fitted = False

        logger.info(
            f"DayTradingClassifier initialized with threshold={threshold}, "
            f"n_estimators={n_estimators}, max_depth={max_depth}"
        )

    def create_target(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        Create binary target: 1 if price increases by threshold, else 0

        Args:
            df: DataFrame with price data
            price_col: Name of price column

        Returns:
            Binary target series
        """
        try:
            # Calculate future return
            future_return = df[price_col].shift(-1) / df[price_col] - 1

            # Create binary target
            target = (future_return > self.threshold).astype(int)

            logger.info(
                f"Created target: {target.sum()} positive samples "
                f"({target.mean():.2%} positive rate)"
            )
            return target

        except Exception as e:
            logger.error(f"Error creating target: {str(e)}")
            raise

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'target',
                        exclude_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            exclude_cols: Columns to exclude from features

        Returns:
            Tuple of (features, target)
        """
        try:
            if exclude_cols is None:
                exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']

            # Get target
            y = df[target_col]

            # Get features (exclude target and specified columns)
            X = df.drop(columns=[target_col] + [col for col in exclude_cols if col in df.columns])

            # Remove any remaining non-numeric columns
            X = X.select_dtypes(include=[np.number])

            # Handle NaN values
            X = X.fillna(X.mean())

            self.feature_names = X.columns.tolist()

            logger.info(f"Prepared features: {X.shape[1]} features, {len(y)} samples")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def train(self, X: pd.DataFrame, y: pd.Series,
             test_size: float = 0.2, early_stopping_rounds: int = 10) -> dict:
        """
        Train the classifier

        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Test set size ratio
            early_stopping_rounds: Early stopping rounds

        Returns:
            Dictionary with training metrics
        """
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            logger.info(f"Training classifier on {len(X_train)} samples...")

            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )

            self.is_fitted = True

            # Evaluate
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba),
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            }

            logger.info(
                f"Training completed - Accuracy: {metrics['accuracy']:.4f}, "
                f"ROC-AUC: {metrics['roc_auc']:.4f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            raise

    def predict_signal(self, X: pd.DataFrame, buy_threshold: float = 0.6,
                      sell_threshold: float = 0.4) -> str:
        """
        Predict trading signal

        Args:
            X: Feature DataFrame (single row or multiple rows)
            buy_threshold: Probability threshold for BUY signal
            sell_threshold: Probability threshold for SELL signal

        Returns:
            Trading signal ('BUY', 'SELL', 'HOLD')
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        try:
            # Get probability of positive class
            proba = self.model.predict_proba(X)[:, 1]

            # Use last prediction if multiple rows
            last_proba = proba[-1]

            if last_proba > buy_threshold:
                signal = 'BUY'
            elif last_proba < sell_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            logger.info(f"Prediction: {signal} (confidence: {last_proba:.4f})")
            return signal

        except Exception as e:
            logger.error(f"Error predicting signal: {str(e)}")
            return 'HOLD'

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Feature DataFrame

        Returns:
            Array of probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        try:
            probas = self.model.predict_proba(X)[:, 1]
            return probas

        except Exception as e:
            logger.error(f"Error predicting probabilities: {str(e)}")
            raise

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        try:
            importance = self.model.feature_importances_

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)

            logger.info(f"Top {top_n} important features:")
            for _, row in importance_df.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

            return importance_df

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate model on test data

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        try:
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'roc_auc': roc_auc_score(y, y_pred_proba),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist()
            }

            logger.info(f"Evaluation metrics:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, filepath: str):
        """
        Save model to disk

        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'threshold': self.threshold
                }, f)

            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath: str):
        """
        Load model from disk

        Args:
            filepath: Path to model file
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.model = data['model']
            self.feature_names = data['feature_names']
            self.threshold = data['threshold']
            self.is_fitted = True

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


class RandomForestSignalGenerator:
    """Random Forest classifier for trading signals (alternative to XGBoost)"""

    def __init__(self, threshold: float = 0.005, n_estimators: int = 100,
                 max_depth: int = 10):
        """
        Initialize Random Forest classifier

        Args:
            threshold: Price change threshold
            n_estimators: Number of trees
            max_depth: Maximum tree depth
        """
        self.threshold = threshold
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None
        self.is_fitted = False

        logger.info(
            f"RandomForestSignalGenerator initialized with "
            f"n_estimators={n_estimators}, max_depth={max_depth}"
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train the classifier"""
        try:
            self.feature_names = X.columns.tolist()

            logger.info(f"Training Random Forest on {len(X)} samples...")
            self.model.fit(X, y)
            self.is_fitted = True

            # Cross-validation score
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')

            metrics = {
                'cv_mean_auc': cv_scores.mean(),
                'cv_std_auc': cv_scores.std()
            }

            logger.info(
                f"Training completed - CV ROC-AUC: {metrics['cv_mean_auc']:.4f} "
                f"(±{metrics['cv_std_auc']:.4f})"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error training Random Forest: {str(e)}")
            raise

    def predict_signal(self, X: pd.DataFrame, buy_threshold: float = 0.6,
                      sell_threshold: float = 0.4) -> str:
        """Predict trading signal"""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        proba = self.model.predict_proba(X)[:, 1][-1]

        if proba > buy_threshold:
            return 'BUY'
        elif proba < sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'


if __name__ == "__main__":
    # Example usage
    import logging.config
    from ml_trading_bot.config.settings import LOGGING_CONFIG
    from ml_trading_bot.data.ingestion import StockDataFetcher
    from ml_trading_bot.features import FeatureEngineer

    logging.config.dictConfig(LOGGING_CONFIG)

    # Fetch and prepare data
    print("Fetching data...")
    fetcher = StockDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', '2022-01-01', '2024-01-01')

    # Engineer features
    print("Engineering features...")
    engineer = FeatureEngineer(df)
    features_df = engineer.add_all_indicators().get_features()

    # Create classifier
    print("\nCreating classifier...")
    classifier = DayTradingClassifier(threshold=0.005)

    # Create target
    features_df['target'] = classifier.create_target(features_df)

    # Prepare features
    X, y = classifier.prepare_features(features_df)

    # Train
    print("\nTraining classifier...")
    metrics = classifier.train(X, y)
    print(f"Training metrics: {metrics}")

    # Feature importance
    print("\nFeature importance:")
    importance_df = classifier.get_feature_importance(top_n=10)
    print(importance_df)

    # Test prediction
    print("\nTesting signal prediction...")
    test_sample = X.tail(1)
    signal = classifier.predict_signal(test_sample)
    print(f"Signal: {signal}")
