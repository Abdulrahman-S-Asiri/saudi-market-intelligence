"""
Advanced LSTM Model for High-Accuracy Stock Prediction

This module implements a state-of-the-art Bidirectional LSTM architecture
with Multi-Head Attention, Residual Connections, and Uncertainty Estimation.

Architecture:
    Input → BatchNorm → BiLSTM(128) → Residual+LayerNorm
          → BiLSTM(256) → Residual+LayerNorm
          → BiLSTM(128) → MultiHeadAttention(4 heads)
          → FC(256→128→64→32) → Output(prediction, uncertainty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for capturing long-range dependencies.

    Uses 4 attention heads with 64 dimensions each for a total of 256 dimensions.
    Includes learnable positional encoding and dropout for regularization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_positional_encoding = use_positional_encoding

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Positional encoding (learnable)
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            output: Attended output of shape (batch, seq_len, embed_dim)
            attention_weights: Attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        if self.use_positional_encoding:
            x = x + self.pos_encoding[:, :seq_len, :]

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(attn_output)

        return output, attn_weights


class ResidualBiLSTMBlock(nn.Module):
    """
    Bidirectional LSTM block with residual connection and layer normalization.

    The residual connection allows gradients to flow more easily through deep networks,
    while layer normalization provides stable training dynamics.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.2,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual
        self.hidden_size = hidden_size

        # Bidirectional LSTM doubles the output size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Projection layer for residual connection if dimensions don't match
        if use_residual and input_size != hidden_size * 2:
            self.residual_proj = nn.Linear(input_size, hidden_size * 2)
        else:
            self.residual_proj = None

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state

        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_size * 2)
            hidden: Final hidden state
        """
        residual = x

        # Apply LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Add residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            lstm_out = lstm_out + residual

        # Apply layer normalization
        output = self.layer_norm(lstm_out)

        return output, hidden


class UncertaintyHead(nn.Module):
    """
    Output head that produces both prediction and uncertainty estimates.

    Uses a dual-output architecture where one branch predicts the value
    and another branch predicts the log variance (uncertainty).
    """

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Prediction head
        self.pred_head = nn.Linear(hidden_size, 1)

        # Uncertainty head (log variance for numerical stability)
        self.uncertainty_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, input_size)

        Returns:
            prediction: Predicted value of shape (batch, 1)
            uncertainty: Uncertainty estimate (log variance) of shape (batch, 1)
        """
        shared_features = self.shared(x)

        prediction = self.pred_head(shared_features)
        log_variance = self.uncertainty_head(shared_features)

        # Clamp log variance for numerical stability
        log_variance = torch.clamp(log_variance, min=-10, max=10)

        return prediction, log_variance


class AdvancedStockLSTM(nn.Module):
    """
    Advanced Bidirectional LSTM for Stock Price Prediction with Uncertainty.

    This architecture combines several state-of-the-art techniques:
    - Bidirectional LSTM layers for capturing temporal patterns in both directions
    - Multi-Head Self-Attention for capturing long-range dependencies
    - Residual connections for better gradient flow
    - Layer Normalization for stable training
    - GELU activation for smoother gradients
    - Uncertainty estimation for confidence-aware predictions

    Architecture:
        Input (batch, seq_len, num_features)
            ↓
        BatchNorm1d
            ↓
        BiLSTM Layer 1 (128 hidden) + Residual + LayerNorm
            ↓
        BiLSTM Layer 2 (256 hidden) + Residual + LayerNorm
            ↓
        BiLSTM Layer 3 (128 hidden) + Residual + LayerNorm
            ↓
        Multi-Head Attention (4 heads)
            ↓
        Global Average Pooling + Last Hidden State Concatenation
            ↓
        FC Block: 512 → 256 → 128 → 64 → 32
            ↓
        Uncertainty Head → (prediction, log_variance)
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: list = [128, 256, 128],
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        use_residual: bool = True,
        use_attention: bool = True
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_sizes = hidden_sizes
        self.use_attention = use_attention

        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(num_features)

        # Build BiLSTM layers with residual connections
        self.lstm_blocks = nn.ModuleList()
        input_size = num_features

        for hidden_size in hidden_sizes:
            block = ResidualBiLSTMBlock(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
                use_residual=use_residual
            )
            self.lstm_blocks.append(block)
            input_size = hidden_size * 2  # BiLSTM doubles output

        # Final LSTM output dimension
        lstm_output_size = hidden_sizes[-1] * 2

        # Multi-Head Attention
        if use_attention:
            self.attention = MultiHeadAttention(
                embed_dim=lstm_output_size,
                num_heads=num_attention_heads,
                dropout=dropout
            )

        # Feature aggregation: concat global avg pool + last hidden
        fc_input_size = lstm_output_size * 2  # avg pool + last hidden

        # Fully connected layers with GELU activation
        self.fc_block = nn.Sequential(
            nn.Linear(fc_input_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.LayerNorm(128),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
        )

        # Uncertainty-aware output head
        self.output_head = UncertaintyHead(input_size=64, hidden_size=32)

        # Store attention weights for interpretability
        self.attention_weights = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Use orthogonal initialization for LSTM weights
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - prediction: Predicted value (batch, 1)
                - uncertainty: Log variance for uncertainty (batch, 1)
                - confidence: Computed confidence score (batch, 1)
                - attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Input batch normalization (transpose for BatchNorm1d)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.input_bn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, features)

        # Pass through BiLSTM blocks
        for lstm_block in self.lstm_blocks:
            x, _ = lstm_block(x)

        # Apply Multi-Head Attention
        if self.use_attention:
            x, attn_weights = self.attention(x)
            self.attention_weights = attn_weights
        else:
            attn_weights = None

        # Feature aggregation
        # Global average pooling
        avg_pooled = x.mean(dim=1)  # (batch, hidden * 2)

        # Last hidden state
        last_hidden = x[:, -1, :]  # (batch, hidden * 2)

        # Concatenate
        combined = torch.cat([avg_pooled, last_hidden], dim=1)

        # Fully connected layers
        fc_out = self.fc_block(combined)

        # Output with uncertainty
        prediction, log_variance = self.output_head(fc_out)

        # Compute confidence from uncertainty
        # Lower variance = higher confidence
        variance = torch.exp(log_variance)
        confidence = torch.sigmoid(-log_variance)  # Maps high variance to low confidence

        output = {
            'prediction': prediction,
            'uncertainty': log_variance,
            'confidence': confidence,
            'variance': variance
        }

        if return_attention:
            output['attention_weights'] = attn_weights

        return output

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using MC Dropout.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary containing mean prediction, std deviation, and confidence
        """
        self.train()  # Enable dropout for MC sampling

        predictions = []
        uncertainties = []

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                predictions.append(output['prediction'])
                uncertainties.append(output['variance'])

        predictions = torch.stack(predictions, dim=0)
        uncertainties = torch.stack(uncertainties, dim=0)

        # Epistemic uncertainty (model uncertainty) from MC samples
        mean_pred = predictions.mean(dim=0)
        epistemic_std = predictions.std(dim=0)

        # Aleatoric uncertainty (data uncertainty) from model output
        aleatoric_var = uncertainties.mean(dim=0)

        # Total uncertainty
        total_var = epistemic_std ** 2 + aleatoric_var
        total_std = torch.sqrt(total_var)

        # Confidence score (inverse of normalized uncertainty)
        confidence = torch.sigmoid(-torch.log(total_var + 1e-8))

        self.eval()  # Restore eval mode

        return {
            'mean_prediction': mean_pred,
            'epistemic_std': epistemic_std,
            'aleatoric_std': torch.sqrt(aleatoric_var),
            'total_std': total_std,
            'confidence': confidence
        }


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss for uncertainty-aware training.

    This loss function penalizes both prediction error and uncertainty calibration,
    encouraging the model to be confident when correct and uncertain when wrong.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        log_variance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.

        Args:
            prediction: Predicted values (batch, 1)
            target: Target values (batch, 1)
            log_variance: Log variance of predictions (batch, 1)

        Returns:
            Loss value
        """
        # Gaussian NLL: 0.5 * (log(var) + (y - mu)^2 / var)
        precision = torch.exp(-log_variance)
        squared_error = (target - prediction) ** 2

        loss = 0.5 * (log_variance + precision * squared_error)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AdvancedStockPredictor:
    """
    High-level predictor class for training and inference with AdvancedStockLSTM.

    Provides a unified interface for:
    - Model training with early stopping
    - Prediction with uncertainty estimates
    - Model saving/loading
    - Performance metrics calculation
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: list = [128, 256, 128],
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 0.0005,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = AdvancedStockLSTM(
            num_features=num_features,
            hidden_sizes=hidden_sizes,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        ).to(self.device)

        self.optimizer = None
        self.scheduler = None
        self.criterion = GaussianNLLLoss()
        self.learning_rate = learning_rate

        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Initial restart period
            T_mult=2,  # Period multiplier after each restart
            eta_min=1e-6
        )

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 200,
        patience: int = 20,
        label_smoothing: float = 0.1
    ) -> Dict[str, list]:
        """
        Train the model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            label_smoothing: Label smoothing factor

        Returns:
            Training history dictionary
        """
        self._setup_optimizer()

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        logger.info(f"Training on {self.device}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            n_train = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()

                # Apply label smoothing
                if label_smoothing > 0:
                    noise = torch.randn_like(batch_y) * label_smoothing
                    batch_y = batch_y + noise

                self.optimizer.zero_grad()

                output = self.model(batch_x)
                loss = self.criterion(
                    output['prediction'],
                    batch_y,
                    output['uncertainty']
                )

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item() * batch_x.size(0)
                train_mae += F.l1_loss(output['prediction'], batch_y).item() * batch_x.size(0)
                n_train += batch_x.size(0)

            self.scheduler.step()

            train_loss /= n_train
            train_mae /= n_train

            # Validation phase
            val_loss, val_mae = self._validate(val_loader)

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_mae'].append(train_mae)
            self.training_history['val_mae'].append(val_mae)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Val MAE={val_mae:.4f}, LR={self.scheduler.get_last_lr()[0]:.6f}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.training_history

    def _validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Validate the model on validation set."""
        self.model.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()

                output = self.model(batch_x)
                loss = self.criterion(
                    output['prediction'],
                    batch_y,
                    output['uncertainty']
                )

                val_loss += loss.item() * batch_x.size(0)
                val_mae += F.l1_loss(output['prediction'], batch_y).item() * batch_x.size(0)
                n_val += batch_x.size(0)

        return val_loss / n_val, val_mae / n_val

    def predict(
        self,
        x: np.ndarray,
        return_confidence: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with optional uncertainty estimation.

        Args:
            x: Input data of shape (n_samples, seq_len, num_features)
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary with predictions and optional confidence scores
        """
        self.model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

            if len(x_tensor.shape) == 2:
                x_tensor = x_tensor.unsqueeze(0)

            output = self.model(x_tensor)

            result = {
                'prediction': output['prediction'].cpu().numpy(),
                'direction': (output['prediction'] > 0).float().cpu().numpy()
            }

            if return_confidence:
                result['confidence'] = output['confidence'].cpu().numpy()
                result['uncertainty'] = output['variance'].cpu().numpy()

            return result

    def predict_with_mc_dropout(
        self,
        x: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions using Monte Carlo Dropout for better uncertainty estimates.

        Args:
            x: Input data
            n_samples: Number of MC samples

        Returns:
            Dictionary with mean prediction, standard deviation, and confidence
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        if len(x_tensor.shape) == 2:
            x_tensor = x_tensor.unsqueeze(0)

        output = self.model.predict_with_uncertainty(x_tensor, n_samples)

        return {
            'prediction': output['mean_prediction'].cpu().numpy(),
            'std': output['total_std'].cpu().numpy(),
            'confidence': output['confidence'].cpu().numpy()
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'model_config': {
                'num_features': self.model.num_features,
                'hidden_sizes': self.model.hidden_sizes
            }
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        logger.info(f"Model loaded from {path}")


def create_ensemble(
    num_features: int,
    n_models: int = 5,
    hidden_sizes: list = [128, 256, 128],
    **kwargs
) -> list:
    """
    Create an ensemble of AdvancedStockLSTM models with different random seeds.

    Args:
        num_features: Number of input features
        n_models: Number of models in ensemble
        hidden_sizes: Hidden layer sizes
        **kwargs: Additional model arguments

    Returns:
        List of AdvancedStockPredictor instances
    """
    ensemble = []

    for i in range(n_models):
        # Set different random seed for each model
        torch.manual_seed(42 + i * 100)
        np.random.seed(42 + i * 100)

        predictor = AdvancedStockPredictor(
            num_features=num_features,
            hidden_sizes=hidden_sizes,
            **kwargs
        )
        ensemble.append(predictor)

    return ensemble


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple AdvancedStockLSTM models.

    Uses different random seeds and aggregates predictions for more robust
    and reliable predictions with better uncertainty calibration.
    """

    def __init__(
        self,
        num_features: int,
        n_models: int = 5,
        hidden_sizes: list = [128, 256, 128],
        **kwargs
    ):
        self.n_models = n_models
        self.models = create_ensemble(
            num_features=num_features,
            n_models=n_models,
            hidden_sizes=hidden_sizes,
            **kwargs
        )

    def train_all(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        **train_kwargs
    ):
        """Train all models in the ensemble."""
        histories = []

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{self.n_models}")
            history = model.train(train_loader, val_loader, **train_kwargs)
            histories.append(history)

        return histories

    def predict(
        self,
        x: np.ndarray,
        aggregation: str = 'mean'
    ) -> Dict[str, np.ndarray]:
        """
        Make ensemble predictions.

        Args:
            x: Input data
            aggregation: How to aggregate predictions ('mean' or 'median')

        Returns:
            Aggregated predictions with uncertainty
        """
        predictions = []
        confidences = []

        for model in self.models:
            result = model.predict(x, return_confidence=True)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])

        predictions = np.array(predictions)
        confidences = np.array(confidences)

        if aggregation == 'mean':
            ensemble_pred = predictions.mean(axis=0)
        else:
            ensemble_pred = np.median(predictions, axis=0)

        # Ensemble uncertainty from disagreement
        ensemble_std = predictions.std(axis=0)

        # Combined confidence
        ensemble_confidence = (confidences.mean(axis=0) *
                               np.exp(-ensemble_std))

        return {
            'prediction': ensemble_pred,
            'std': ensemble_std,
            'confidence': ensemble_confidence,
            'direction': (ensemble_pred > 0).astype(float),
            'all_predictions': predictions
        }


if __name__ == "__main__":
    # Quick test of the model
    logging.basicConfig(level=logging.INFO)

    # Create dummy data
    batch_size = 32
    seq_len = 60
    num_features = 35

    x = torch.randn(batch_size, seq_len, num_features)

    # Test model
    model = AdvancedStockLSTM(
        num_features=num_features,
        hidden_sizes=[128, 256, 128],
        num_attention_heads=4
    )

    output = model(x, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Prediction shape: {output['prediction'].shape}")
    print(f"Confidence shape: {output['confidence'].shape}")
    print(f"Attention weights shape: {output['attention_weights'].shape}")

    print("\nAdvanced LSTM Model Test: OK")
