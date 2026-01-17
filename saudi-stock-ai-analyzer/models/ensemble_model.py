"""
Ensemble model combining LSTM with XGBoost for robust predictions
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import os
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import StockPredictor, ModelCache
from utils.config import MODEL_SAVE_PATH


class XGBoostPredictor:
    """XGBoost model for stock prediction"""

    def __init__(self, **kwargs):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        self.model = xgb.XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.05),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.8),
            min_child_weight=kwargs.get('min_child_weight', 3),
            reg_alpha=kwargs.get('reg_alpha', 0.1),
            reg_lambda=kwargs.get('reg_lambda', 1.0),
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        self.is_trained = False
        self.feature_importance = None

    def prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Flatten sequence data for XGBoost"""
        if len(X.shape) == 3:
            batch_size = X.shape[0]
            return X.reshape(batch_size, -1)
        return X

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: bool = True
    ) -> Dict:
        """Train XGBoost model"""
        X_train_flat = self.prepare_features(X_train)

        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_flat = self.prepare_features(X_val)
            eval_set = [(X_val_flat, y_val)]

        self.model.fit(
            X_train_flat,
            y_train,
            eval_set=eval_set,
            verbose=False
        )

        self.is_trained = True
        self.feature_importance = self.model.feature_importances_

        return {'trained': True}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_flat = self.prepare_features(X)
        return self.model.predict(X_flat)

    def save(self, filepath: str):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data['model']
            self.is_trained = data['is_trained']
            self.feature_importance = data.get('feature_importance')
            return True
        return False


class EnsemblePredictor:
    """
    Ensemble model combining LSTM and XGBoost predictions
    Weights are dynamically adjusted based on recent accuracy
    """

    def __init__(
        self,
        input_size: int,
        lstm_weight: float = 0.6,
        xgb_weight: float = 0.4,
        use_dynamic_weights: bool = True
    ):
        self.input_size = input_size
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight
        self.use_dynamic_weights = use_dynamic_weights

        # Initialize models
        self.lstm_predictor = StockPredictor(input_size=input_size, use_attention=True)

        self.xgb_predictor = None
        if HAS_XGBOOST:
            self.xgb_predictor = XGBoostPredictor()

        # Track prediction accuracy
        self.prediction_history = {
            'lstm': {'predictions': [], 'actuals': [], 'errors': []},
            'xgb': {'predictions': [], 'actuals': [], 'errors': []},
            'ensemble': {'predictions': [], 'actuals': [], 'errors': []}
        }

        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """Train both LSTM and XGBoost models"""
        results = {}

        # Train LSTM
        if verbose:
            print("Training LSTM model...")
        lstm_history = self.lstm_predictor.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, patience=patience, verbose=verbose
        )
        results['lstm'] = lstm_history

        # Train XGBoost
        if self.xgb_predictor:
            if verbose:
                print("\nTraining XGBoost model...")
            xgb_result = self.xgb_predictor.train(X_train, y_train, X_val, y_val, verbose=verbose)
            results['xgb'] = xgb_result

        # Calculate initial weights based on validation performance
        if X_val is not None and y_val is not None and len(X_val) > 0:
            self._update_weights(X_val, y_val)

        self.is_trained = True
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        lstm_pred = self.lstm_predictor.predict(X)

        if self.xgb_predictor and self.xgb_predictor.is_trained:
            xgb_pred = self.xgb_predictor.predict(X)
            ensemble_pred = self.lstm_weight * lstm_pred + self.xgb_weight * xgb_pred
        else:
            ensemble_pred = lstm_pred

        return ensemble_pred

    def predict_with_breakdown(self, X: np.ndarray) -> Dict:
        """Get predictions from each model"""
        lstm_pred = self.lstm_predictor.predict(X)

        result = {
            'lstm': lstm_pred,
            'lstm_weight': self.lstm_weight,
            'ensemble': lstm_pred.copy()
        }

        if self.xgb_predictor and self.xgb_predictor.is_trained:
            xgb_pred = self.xgb_predictor.predict(X)
            result['xgb'] = xgb_pred
            result['xgb_weight'] = self.xgb_weight
            result['ensemble'] = self.lstm_weight * lstm_pred + self.xgb_weight * xgb_pred

        return result

    def predict_direction(self, X: np.ndarray, current_price: float) -> Tuple[str, float, Dict]:
        """Predict price direction with ensemble"""
        breakdown = self.predict_with_breakdown(X)
        ensemble_pred = breakdown['ensemble'][0]

        price_change = ensemble_pred - current_price
        percent_change = (price_change / current_price) * 100 if current_price != 0 else 0

        # Determine direction
        if percent_change > 0.5:
            direction = "UP"
        elif percent_change < -0.5:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        # Confidence based on model agreement and magnitude
        confidence = self._calculate_confidence(breakdown, current_price)

        return direction, confidence, breakdown

    def _calculate_confidence(self, breakdown: Dict, current_price: float) -> float:
        """Calculate confidence based on model agreement"""
        lstm_pred = breakdown['lstm'][0]
        lstm_direction = 1 if lstm_pred > current_price else -1

        confidence = 0.5  # Base confidence

        if 'xgb' in breakdown:
            xgb_pred = breakdown['xgb'][0]
            xgb_direction = 1 if xgb_pred > current_price else -1

            # Agreement bonus
            if lstm_direction == xgb_direction:
                confidence += 0.2

            # Magnitude factor
            avg_change = abs((breakdown['ensemble'][0] - current_price) / current_price)
            confidence += min(avg_change * 10, 0.3)
        else:
            avg_change = abs((lstm_pred - current_price) / current_price)
            confidence += min(avg_change * 10, 0.3)

        return min(max(confidence, 0.0), 1.0)

    def _update_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Update weights based on validation performance"""
        if not self.use_dynamic_weights:
            return

        lstm_pred = self.lstm_predictor.predict(X_val)
        lstm_mse = np.mean((lstm_pred - y_val) ** 2)

        if self.xgb_predictor and self.xgb_predictor.is_trained:
            xgb_pred = self.xgb_predictor.predict(X_val)
            xgb_mse = np.mean((xgb_pred - y_val) ** 2)

            # Calculate weights inversely proportional to error
            total_error = lstm_mse + xgb_mse
            if total_error > 0:
                self.lstm_weight = 1 - (lstm_mse / total_error)
                self.xgb_weight = 1 - (xgb_mse / total_error)

                # Normalize
                total_weight = self.lstm_weight + self.xgb_weight
                self.lstm_weight /= total_weight
                self.xgb_weight /= total_weight

    def update_accuracy(self, predictions: Dict, actuals: np.ndarray):
        """Update accuracy tracking for dynamic weight adjustment"""
        for model_name in ['lstm', 'xgb', 'ensemble']:
            if model_name in predictions:
                pred = predictions[model_name]
                error = np.abs(pred - actuals)
                self.prediction_history[model_name]['predictions'].extend(pred.tolist())
                self.prediction_history[model_name]['actuals'].extend(actuals.tolist())
                self.prediction_history[model_name]['errors'].extend(error.tolist())

    def get_model_statistics(self) -> Dict:
        """Get statistics about model performance"""
        stats = {}

        for model_name, history in self.prediction_history.items():
            if history['errors']:
                recent_errors = history['errors'][-20:]  # Last 20 predictions
                stats[model_name] = {
                    'mean_error': np.mean(recent_errors),
                    'std_error': np.std(recent_errors),
                    'n_predictions': len(history['predictions'])
                }

        stats['weights'] = {
            'lstm': self.lstm_weight,
            'xgb': self.xgb_weight if self.xgb_predictor else 0
        }

        return stats

    def save(self, symbol: str):
        """Save ensemble models"""
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        # Save LSTM
        lstm_path = os.path.join(MODEL_SAVE_PATH, f"ensemble_lstm_{symbol}.pth")
        self.lstm_predictor.save(lstm_path)

        # Save XGBoost
        if self.xgb_predictor and self.xgb_predictor.is_trained:
            xgb_path = os.path.join(MODEL_SAVE_PATH, f"ensemble_xgb_{symbol}.joblib")
            self.xgb_predictor.save(xgb_path)

        # Save ensemble metadata
        meta_path = os.path.join(MODEL_SAVE_PATH, f"ensemble_meta_{symbol}.joblib")
        joblib.dump({
            'lstm_weight': self.lstm_weight,
            'xgb_weight': self.xgb_weight,
            'prediction_history': self.prediction_history,
            'is_trained': self.is_trained
        }, meta_path)

    def load(self, symbol: str) -> bool:
        """Load ensemble models"""
        try:
            # Load LSTM
            lstm_path = os.path.join(MODEL_SAVE_PATH, f"ensemble_lstm_{symbol}.pth")
            self.lstm_predictor.load(lstm_path)

            # Load XGBoost
            if self.xgb_predictor:
                xgb_path = os.path.join(MODEL_SAVE_PATH, f"ensemble_xgb_{symbol}.joblib")
                if os.path.exists(xgb_path):
                    self.xgb_predictor.load(xgb_path)

            # Load metadata
            meta_path = os.path.join(MODEL_SAVE_PATH, f"ensemble_meta_{symbol}.joblib")
            if os.path.exists(meta_path):
                meta = joblib.load(meta_path)
                self.lstm_weight = meta.get('lstm_weight', 0.6)
                self.xgb_weight = meta.get('xgb_weight', 0.4)
                self.prediction_history = meta.get('prediction_history', self.prediction_history)
                self.is_trained = meta.get('is_trained', True)

            return True
        except Exception as e:
            print(f"Failed to load ensemble model: {e}")
            return False


if __name__ == "__main__":
    print("Testing Ensemble Model...")
    print("=" * 60)

    # Create dummy data
    n_samples = 300
    sequence_length = 60
    n_features = 8

    X_dummy = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
    y_dummy = np.random.randn(n_samples).astype(np.float32)

    # Split
    train_idx = int(0.7 * n_samples)
    val_idx = int(0.85 * n_samples)

    X_train = X_dummy[:train_idx]
    X_val = X_dummy[train_idx:val_idx]
    X_test = X_dummy[val_idx:]
    y_train = y_dummy[:train_idx]
    y_val = y_dummy[train_idx:val_idx]
    y_test = y_dummy[val_idx:]

    # Create and train ensemble
    ensemble = EnsemblePredictor(input_size=n_features)
    print(f"\nTraining ensemble model...")
    results = ensemble.train(X_train, y_train, X_val, y_val, epochs=20, patience=5)

    # Make predictions
    breakdown = ensemble.predict_with_breakdown(X_test[:5])
    print(f"\nPrediction breakdown:")
    print(f"  LSTM predictions: {breakdown['lstm'][:5]}")
    if 'xgb' in breakdown:
        print(f"  XGBoost predictions: {breakdown['xgb'][:5]}")
    print(f"  Ensemble predictions: {breakdown['ensemble'][:5]}")
    print(f"  Weights - LSTM: {ensemble.lstm_weight:.2f}, XGBoost: {ensemble.xgb_weight:.2f}")

    # Direction prediction
    direction, confidence, _ = ensemble.predict_direction(X_test[:1], current_price=0.5)
    print(f"\nDirection: {direction}, Confidence: {confidence:.2%}")

    print("\nEnsemble model test completed!")
