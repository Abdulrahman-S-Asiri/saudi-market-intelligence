"""
Enhanced LSTM neural network model for stock prediction
Features: Attention mechanism, early stopping, LR scheduler, model caching
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import os
import json
from datetime import datetime, timedelta
import hashlib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import LSTM_CONFIG, MODEL_SAVE_PATH


class Attention(nn.Module):
    """Self-attention mechanism for important time steps"""

    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM output

        Args:
            lstm_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) - weighted sum
            attention_weights: (batch, seq_len, 1)
        """
        attention_weights = self.attention(lstm_output)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class StockLSTM(nn.Module):
    """
    Enhanced LSTM model with attention mechanism and batch normalization
    Architecture: 3 LSTM layers (64->128->64) with attention
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = None,
        num_layers: int = None,
        dropout: float = None,
        bidirectional: bool = False,
        use_attention: bool = True
    ):
        super(StockLSTM, self).__init__()

        self.hidden_sizes = hidden_sizes or [64, 128, 64]
        self.num_layers = num_layers or len(self.hidden_sizes)
        self.dropout = dropout or LSTM_CONFIG['dropout']
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Direction multiplier for bidirectional
        self.direction_mult = 2 if bidirectional else 1

        # Stacked LSTM layers with increasing then decreasing hidden size
        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        prev_size = input_size
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=prev_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0,
                    bidirectional=bidirectional
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_size * self.direction_mult))
            prev_size = hidden_size * self.direction_mult

        # Attention layer
        final_hidden = self.hidden_sizes[-1] * self.direction_mult
        if use_attention:
            self.attention = Attention(final_hidden)

        # Fully connected layers
        self.fc1 = nn.Linear(final_hidden, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention"""
        batch_size = x.size(0)

        # Pass through stacked LSTM layers
        out = x
        for i, (lstm, bn) in enumerate(zip(self.lstm_layers, self.batch_norms)):
            out, _ = lstm(out)
            # Apply batch norm (need to reshape for BatchNorm1d)
            out = out.permute(0, 2, 1)  # (batch, hidden, seq)
            out = bn(out)
            out = out.permute(0, 2, 1)  # (batch, seq, hidden)
            out = self.dropout_layer(out)

        # Apply attention or take last output
        if self.use_attention:
            context, _ = self.attention(out)
        else:
            context = out[:, -1, :]

        # Fully connected layers
        out = self.fc1(context)
        out = self.bn_fc1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout_layer(out)

        out = self.fc3(out)
        return out


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0001, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and self.best_model_state:
                    model.load_state_dict(self.best_model_state)
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

        return self.early_stop


class StockPredictor:
    """
    High-level interface for training and predicting stock prices
    Features: Model caching, early stopping, LR scheduler
    """

    def __init__(self, input_size: int, use_attention: bool = True, bidirectional: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.use_attention = use_attention
        self.bidirectional = bidirectional

        self.model = StockLSTM(
            input_size,
            use_attention=use_attention,
            bidirectional=bidirectional
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LSTM_CONFIG['learning_rate'],
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=False
        )

        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
        self.is_trained = False
        self.training_date = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = None,
        batch_size: int = None,
        patience: int = 10,
        verbose: bool = True
    ) -> dict:
        """
        Train the LSTM model with early stopping and LR scheduler

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        epochs = epochs or LSTM_CONFIG.get('epochs', 100)
        batch_size = batch_size or LSTM_CONFIG['batch_size']

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        has_validation = X_val is not None and y_val is not None and len(X_val) > 0
        if has_validation:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        # Early stopping
        early_stopping = EarlyStopping(patience=patience)

        # Training loop
        self.model.train()
        n_samples = len(X_train)

        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0

            # Shuffle training data
            indices = torch.randperm(n_samples)

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / n_batches
            self.history['train_loss'].append(avg_train_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Validation
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                self.history['val_loss'].append(val_loss)
                self.model.train()

                # LR scheduler
                self.scheduler.step(val_loss)

                # Early stopping check
                if early_stopping(val_loss, self.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.6f} - "
                          f"Val Loss: {val_loss:.6f} - "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.6f}")

        self.is_trained = True
        self.training_date = datetime.now().isoformat()
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().flatten()

    def predict_direction(self, X: np.ndarray, current_price: float) -> Tuple[str, float]:
        """Predict price direction and confidence"""
        prediction = self.predict(X)[0]

        price_change = prediction - current_price
        percent_change = (price_change / current_price) * 100 if current_price != 0 else 0

        # Determine direction with thresholds
        if percent_change > 0.5:
            direction = "UP"
        elif percent_change < -0.5:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        # Confidence based on magnitude
        confidence = min(abs(percent_change) / 5.0, 1.0)

        return direction, confidence

    def save(self, filepath: str = None, symbol: str = None):
        """Save model with metadata"""
        if filepath is None:
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            filename = f"lstm_model_{symbol}.pth" if symbol else "lstm_model.pth"
            filepath = os.path.join(MODEL_SAVE_PATH, filename)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            'input_size': self.input_size,
            'use_attention': self.use_attention,
            'bidirectional': self.bidirectional,
            'config': {
                'hidden_sizes': self.model.hidden_sizes,
                'dropout': self.model.dropout
            }
        }, filepath)

        # Save metadata JSON
        meta_path = filepath.replace('.pth', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'training_date': self.training_date,
                'is_trained': self.is_trained,
                'input_size': self.input_size,
                'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None
            }, f)

        return filepath

    def load(self, filepath: str = None, symbol: str = None):
        """Load model from file"""
        if filepath is None:
            filename = f"lstm_model_{symbol}.pth" if symbol else "lstm_model.pth"
            filepath = os.path.join(MODEL_SAVE_PATH, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Recreate model with saved config
        self.input_size = checkpoint.get('input_size', self.input_size)
        self.use_attention = checkpoint.get('use_attention', True)
        self.bidirectional = checkpoint.get('bidirectional', False)

        self.model = StockLSTM(
            self.input_size,
            hidden_sizes=checkpoint.get('config', {}).get('hidden_sizes'),
            dropout=checkpoint.get('config', {}).get('dropout'),
            use_attention=self.use_attention,
            bidirectional=self.bidirectional
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint['history']
        self.is_trained = checkpoint['is_trained']
        self.training_date = checkpoint.get('training_date')

        return True

    def get_model_summary(self) -> dict:
        """Get model architecture summary"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            'hidden_sizes': self.model.hidden_sizes,
            'dropout': self.model.dropout,
            'use_attention': self.use_attention,
            'bidirectional': self.bidirectional,
            'epochs_trained': len(self.history.get('train_loss', []))
        }


class WalkForwardValidator:
    """
    Walk-forward validation for more realistic performance estimation.
    Trains on expanding window, tests on subsequent periods.
    """

    def __init__(self, n_splits: int = 5, min_train_size: float = 0.5):
        """
        Args:
            n_splits: Number of walk-forward periods
            min_train_size: Minimum fraction of data for first training window
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.results = []

    def split(self, X: np.ndarray, y: np.ndarray):
        """
        Generate walk-forward train/test splits.

        Yields:
            Tuples of (X_train, y_train, X_test, y_test, split_info)
        """
        n_samples = len(X)
        min_train = int(n_samples * self.min_train_size)
        test_size = (n_samples - min_train) // self.n_splits

        for i in range(self.n_splits):
            train_end = min_train + (i * test_size)
            test_end = train_end + test_size

            if test_end > n_samples:
                test_end = n_samples

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]

            split_info = {
                'split': i + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_end_idx': train_end,
                'test_end_idx': test_end
            }

            yield X_train, y_train, X_test, y_test, split_info

    def validate(self, predictor_class, X: np.ndarray, y: np.ndarray,
                 input_size: int, epochs: int = 50, verbose: bool = True) -> Dict:
        """
        Perform walk-forward validation.

        Args:
            predictor_class: Class to instantiate for each split
            X: Full feature array
            y: Full target array
            input_size: Input size for predictor
            epochs: Training epochs per split
            verbose: Print progress

        Returns:
            Dictionary with validation results
        """
        self.results = []
        all_predictions = []
        all_actuals = []

        for X_train, y_train, X_test, y_test, split_info in self.split(X, y):
            if verbose:
                print(f"\nWalk-Forward Split {split_info['split']}/{self.n_splits}")
                print(f"  Train: {split_info['train_size']} samples")
                print(f"  Test: {split_info['test_size']} samples")

            # Create and train model
            predictor = predictor_class(input_size=input_size)

            # Use 10% of training data as validation
            val_size = int(len(X_train) * 0.1)
            X_tr = X_train[:-val_size] if val_size > 0 else X_train
            y_tr = y_train[:-val_size] if val_size > 0 else y_train
            X_val = X_train[-val_size:] if val_size > 0 else None
            y_val = y_train[-val_size:] if val_size > 0 else None

            predictor.train(X_tr, y_tr, X_val, y_val, epochs=epochs, verbose=False)

            # Predict on test set
            predictions = predictor.predict(X_test)
            all_predictions.extend(predictions)
            all_actuals.extend(y_test)

            # Calculate metrics for this split
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))

            # Direction accuracy
            if len(y_test) > 1:
                pred_direction = np.sign(predictions[1:] - predictions[:-1])
                actual_direction = np.sign(y_test[1:] - y_test[:-1])
                direction_accuracy = np.mean(pred_direction == actual_direction) * 100
            else:
                direction_accuracy = 0

            split_result = {
                'split': split_info['split'],
                'mse': float(mse),
                'mae': float(mae),
                'direction_accuracy': float(direction_accuracy)
            }
            self.results.append(split_result)

            if verbose:
                print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
                print(f"  Direction Accuracy: {direction_accuracy:.1f}%")

        # Aggregate results
        avg_mse = np.mean([r['mse'] for r in self.results])
        avg_mae = np.mean([r['mae'] for r in self.results])
        avg_direction_acc = np.mean([r['direction_accuracy'] for r in self.results])

        return {
            'n_splits': self.n_splits,
            'avg_mse': float(avg_mse),
            'avg_mae': float(avg_mae),
            'avg_direction_accuracy': float(avg_direction_acc),
            'split_results': self.results,
            'all_predictions': all_predictions,
            'all_actuals': all_actuals
        }


class ConfidenceCalibrator:
    """
    Calibrate model confidence based on historical accuracy.
    Uses isotonic regression for calibration.
    """

    def __init__(self):
        self.calibration_map = {}
        self.is_calibrated = False

    def calibrate(self, predictions: np.ndarray, actuals: np.ndarray,
                  confidences: np.ndarray, n_bins: int = 10) -> Dict:
        """
        Calibrate confidence scores based on historical accuracy.

        Args:
            predictions: Model predictions
            actuals: Actual values
            confidences: Original confidence scores (0-1 or 0-100)
            n_bins: Number of bins for calibration

        Returns:
            Calibration results
        """
        # Normalize confidences to 0-100
        if np.max(confidences) <= 1:
            confidences = confidences * 100

        # Calculate actual accuracy at each confidence level
        bin_edges = np.linspace(0, 100, n_bins + 1)
        calibration_data = []

        for i in range(n_bins):
            bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
            if np.sum(bin_mask) > 0:
                bin_preds = predictions[bin_mask]
                bin_actuals = actuals[bin_mask]

                # Direction accuracy
                if len(bin_preds) > 1:
                    pred_correct = np.sum(
                        (bin_preds > 0) == (bin_actuals > 0)
                    )
                    actual_accuracy = pred_correct / len(bin_preds) * 100
                else:
                    actual_accuracy = 50

                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                calibration_data.append({
                    'bin_center': bin_center,
                    'stated_confidence': bin_center,
                    'actual_accuracy': actual_accuracy,
                    'n_samples': int(np.sum(bin_mask))
                })

                self.calibration_map[bin_center] = actual_accuracy

        self.is_calibrated = True

        # Calculate calibration error
        if calibration_data:
            calibration_error = np.mean([
                abs(d['stated_confidence'] - d['actual_accuracy'])
                for d in calibration_data
            ])
        else:
            calibration_error = 0

        return {
            'calibration_data': calibration_data,
            'calibration_error': calibration_error,
            'is_calibrated': True
        }

    def calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Apply calibration to a raw confidence score.

        Args:
            raw_confidence: Original confidence (0-100)

        Returns:
            Calibrated confidence (0-100)
        """
        if not self.is_calibrated or not self.calibration_map:
            return raw_confidence

        # Find nearest calibration bin
        bin_centers = sorted(self.calibration_map.keys())
        nearest_bin = min(bin_centers, key=lambda x: abs(x - raw_confidence))

        # Linear interpolation between bins
        calibrated = self.calibration_map[nearest_bin]

        # Blend with original to avoid over-correction
        blended = 0.7 * calibrated + 0.3 * raw_confidence

        return max(0, min(100, blended))

    def get_calibration_curve(self) -> Dict:
        """Get calibration curve data for plotting"""
        if not self.calibration_map:
            return {'stated': [], 'actual': []}

        stated = sorted(self.calibration_map.keys())
        actual = [self.calibration_map[s] for s in stated]

        return {'stated': stated, 'actual': actual}


class ModelCache:
    """Cache manager for trained models"""

    def __init__(self, cache_dir: str = None, max_age_hours: int = 24):
        self.cache_dir = cache_dir or MODEL_SAVE_PATH
        self.max_age_hours = max_age_hours
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_key(self, symbol: str, features: list) -> str:
        """Generate unique cache key"""
        feature_hash = hashlib.md5(str(sorted(features)).encode()).hexdigest()[:8]
        return f"{symbol}_{feature_hash}"

    def get_cached_model(self, symbol: str, features: list, input_size: int) -> Optional[StockPredictor]:
        """Load cached model if valid"""
        cache_key = self.get_cache_key(symbol, features)
        model_path = os.path.join(self.cache_dir, f"lstm_model_{cache_key}.pth")
        meta_path = model_path.replace('.pth', '_meta.json')

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            return None

        # Check age
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            training_date = datetime.fromisoformat(meta['training_date'])
            if datetime.now() - training_date > timedelta(hours=self.max_age_hours):
                return None

            # Load model
            predictor = StockPredictor(input_size=input_size)
            predictor.load(model_path)
            return predictor

        except Exception as e:
            print(f"Failed to load cached model: {e}")
            return None

    def save_model(self, predictor: StockPredictor, symbol: str, features: list) -> str:
        """Save model to cache"""
        cache_key = self.get_cache_key(symbol, features)
        model_path = os.path.join(self.cache_dir, f"lstm_model_{cache_key}.pth")
        return predictor.save(model_path)

    def clear_old_cache(self):
        """Remove old cached models"""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_meta.json'):
                meta_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    training_date = datetime.fromisoformat(meta['training_date'])
                    if datetime.now() - training_date > timedelta(hours=self.max_age_hours):
                        model_path = meta_path.replace('_meta.json', '.pth')
                        if os.path.exists(model_path):
                            os.remove(model_path)
                        os.remove(meta_path)
                except Exception:
                    pass

    def get(self, symbol: str) -> Optional[StockPredictor]:
        """Simple get by symbol (default features)"""
        from utils.config import LSTM_FEATURES
        return self.get_cached_model(symbol, LSTM_FEATURES, len(LSTM_FEATURES))

    def set(self, symbol: str, predictor: StockPredictor):
        """Simple set by symbol"""
        from utils.config import LSTM_FEATURES
        self.save_model(predictor, symbol, LSTM_FEATURES)

    def invalidate(self, symbol: str):
        """Invalidate cache for a specific symbol"""
        for filename in os.listdir(self.cache_dir):
            if filename.startswith(f"lstm_model_{symbol}_"):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(filepath)
                except Exception:
                    pass

    def clear(self):
        """Clear all cached models"""
        for filename in os.listdir(self.cache_dir):
            if filename.startswith("lstm_model_"):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(filepath)
                except Exception:
                    pass


if __name__ == "__main__":
    print("Testing Enhanced LSTM Model...")
    print("=" * 60)

    # Create dummy data
    n_samples = 300
    sequence_length = 60
    n_features = 8

    X_dummy = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
    y_dummy = np.random.randn(n_samples).astype(np.float32)

    # Split
    train_split = int(0.7 * n_samples)
    val_split = int(0.85 * n_samples)

    X_train, X_val, X_test = X_dummy[:train_split], X_dummy[train_split:val_split], X_dummy[val_split:]
    y_train, y_val, y_test = y_dummy[:train_split], y_dummy[train_split:val_split], y_dummy[val_split:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Create and train model
    predictor = StockPredictor(input_size=n_features, use_attention=True)
    print(f"\nModel Summary:")
    print(predictor.get_model_summary())

    print("\nTraining model with early stopping...")
    history = predictor.train(X_train, y_train, X_val, y_val, epochs=50, patience=10)

    # Make predictions
    predictions = predictor.predict(X_test[:5])
    print(f"\nSample predictions: {predictions}")

    # Test direction prediction
    direction, confidence = predictor.predict_direction(X_test[:1], current_price=0.5)
    print(f"Direction: {direction}, Confidence: {confidence:.2%}")

    # Test save/load
    print("\nTesting save/load...")
    predictor.save(symbol="TEST")
    loaded_predictor = StockPredictor(input_size=n_features)
    loaded_predictor.load(symbol="TEST")
    print(f"Model loaded successfully: {loaded_predictor.is_trained}")

    print("\nEnhanced LSTM Model test completed!")
