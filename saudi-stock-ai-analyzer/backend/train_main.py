#!/usr/bin/env python3
# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
TFT Training Main Script
========================

Main execution script for training the Temporal Fusion Transformer (TFT)
model on Saudi Stock Market data.

Features:
- Initializes SOTAStockPredictor with proper configuration
- Sets up TradingAwareLoss (0.7 Quantile + 0.3 Sharpe)
- Uses AdamW optimizer with OneCycleLR scheduler
- Implements checkpointing based on validation Sharpe Ratio
- Includes comprehensive shape verification

Usage:
    python train_main.py --stock_code 2222 --epochs 100 --batch_size 32

Author: Claude AI / Abdulrahman Asiri
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_tft import SOTAStockPredictor, TFTConfig
from models.loss_functions import TradingAwareLoss, QuantileLoss, SharpeLoss
from models.training_utils import (
    TFTTrainer,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    EarlyStopping,
    ExponentialMovingAverage
)
from data.data_loader_tft import (
    create_tft_dataloaders,
    TFTFeatureConfig,
    pad_features_to_model_size,
    verify_batch_shapes
)
from data.data_preprocessor import preprocess_stock_data, DataPreprocessor
from data.data_loader import SaudiStockDataLoader
from utils.config import (
    ADVANCED_LSTM_CONFIG,
    MODEL_SAVE_PATH,
    SCALER_SAVE_PATH,
    SAUDI_STOCKS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@torch.no_grad()
def estimate_market_cap(df, stock_code: str) -> float:
    """
    Estimate market cap from price and volume data.

    This is a rough estimate used for market cap grouping.
    For accurate values, use actual market cap data from API.
    """
    if df is None or len(df) == 0:
        return 50e9  # Default to mid-cap

    # Rough estimate: avg_price * avg_volume * 250 (trading days)
    avg_price = df['Close'].mean() if 'Close' in df.columns else 100
    avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 1e6

    # Very rough multiplier for outstanding shares estimation
    # Aramco-like stocks will have much higher actual market cap
    if stock_code == '2222':  # Aramco
        return 2e12  # ~2 trillion SAR
    elif stock_code in ['1120', '1182', '2010']:  # Large caps
        return 100e9  # ~100 billion SAR

    return float(avg_price * avg_volume * 250)


class TFTTrainingConfig:
    """Training configuration for TFT model."""

    def __init__(
        self,
        # Model architecture
        hidden_size: int = 128,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        num_dynamic_features: int = 40,
        num_known_features: int = 5,
        sequence_length: int = 60,
        forecast_horizon: int = 5,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),

        # Training
        batch_size: int = 32,
        epochs: int = 200,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        patience: int = 30,

        # Loss weights (0.7 Quantile + 0.3 Sharpe as specified)
        quantile_weight: float = 0.7,
        sharpe_weight: float = 0.3,
        sortino_weight: float = 0.0,
        mdd_weight: float = 0.0,

        # Scheduler
        scheduler_type: str = 'onecycle',  # 'onecycle' or 'cosine'
        onecycle_pct_start: float = 0.3,
        onecycle_div_factor: float = 25.0,

        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.999,

        # Data
        train_split: float = 0.7,
        val_split: float = 0.15,

        # Checkpointing
        checkpoint_dir: str = None,
        save_every_n_epochs: int = 10,
    ):
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.num_dynamic_features = num_dynamic_features
        self.num_known_features = num_known_features
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.patience = patience

        self.quantile_weight = quantile_weight
        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight
        self.mdd_weight = mdd_weight

        self.scheduler_type = scheduler_type
        self.onecycle_pct_start = onecycle_pct_start
        self.onecycle_div_factor = onecycle_div_factor

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        self.train_split = train_split
        self.val_split = val_split

        self.checkpoint_dir = checkpoint_dir or MODEL_SAVE_PATH
        self.save_every_n_epochs = save_every_n_epochs

    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> 'TFTTrainingConfig':
        """Create config from dictionary."""
        return cls(**d)


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """
    Manages model checkpoints during training.

    Saves checkpoints based on:
    1. Best validation Sharpe Ratio (primary metric)
    2. Best validation loss
    3. Regular interval checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str = 'tft_model',
        max_checkpoints: int = 5
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_sharpe = float('-inf')
        self.best_val_loss = float('inf')
        self.checkpoint_history = []

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_sharpe: float,
        config: TFTTrainingConfig,
        metadata: Dict = None,
        ema_state: Dict = None
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_sharpe: Validation Sharpe ratio
            config: Training configuration
            metadata: Additional metadata
            ema_state: EMA shadow weights if using EMA

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler and hasattr(scheduler, 'state_dict') else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_sharpe': val_sharpe,
            'config': config.to_dict(),
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
        }

        if ema_state is not None:
            checkpoint['ema_state'] = ema_state

        # Determine checkpoint type and filename
        is_best_sharpe = val_sharpe > self.best_sharpe
        is_best_loss = val_loss < self.best_val_loss

        if is_best_sharpe:
            self.best_sharpe = val_sharpe
            filename = f'{self.model_name}_best_sharpe.pt'
            checkpoint['is_best_sharpe'] = True
            logger.info(f"New best Sharpe Ratio: {val_sharpe:.4f}")
        elif is_best_loss:
            self.best_val_loss = val_loss
            filename = f'{self.model_name}_best_loss.pt'
            checkpoint['is_best_loss'] = True
            logger.info(f"New best validation loss: {val_loss:.6f}")
        else:
            filename = f'{self.model_name}_epoch_{epoch:04d}.pt'

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

        # Track checkpoint history
        self.checkpoint_history.append({
            'epoch': epoch,
            'filepath': filepath,
            'val_sharpe': val_sharpe,
            'val_loss': val_loss
        })

        # Clean up old checkpoints (keep best + recent)
        self._cleanup_old_checkpoints()

        return filepath

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping best and most recent."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return

        # Always keep best_sharpe and best_loss
        protected = {
            os.path.join(self.checkpoint_dir, f'{self.model_name}_best_sharpe.pt'),
            os.path.join(self.checkpoint_dir, f'{self.model_name}_best_loss.pt'),
        }

        # Sort by epoch and remove oldest non-protected
        sorted_history = sorted(self.checkpoint_history, key=lambda x: x['epoch'])

        for entry in sorted_history[:-self.max_checkpoints]:
            if entry['filepath'] not in protected and os.path.exists(entry['filepath']):
                os.remove(entry['filepath'])
                self.checkpoint_history.remove(entry)

    def load_checkpoint(self, filepath: str = None, load_best_sharpe: bool = True):
        """
        Load a checkpoint.

        Args:
            filepath: Specific checkpoint to load
            load_best_sharpe: If True and filepath is None, load best Sharpe checkpoint

        Returns:
            Checkpoint dictionary
        """
        if filepath is None:
            if load_best_sharpe:
                filepath = os.path.join(
                    self.checkpoint_dir,
                    f'{self.model_name}_best_sharpe.pt'
                )
            else:
                filepath = os.path.join(
                    self.checkpoint_dir,
                    f'{self.model_name}_best_loss.pt'
                )

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location='cpu')
        logger.info(f"Loaded checkpoint from {filepath}")
        logger.info(f"  Epoch: {checkpoint['epoch']}")
        logger.info(f"  Val Sharpe: {checkpoint.get('val_sharpe', 'N/A')}")
        logger.info(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")

        return checkpoint


# =============================================================================
# CUSTOM TFT TRAINING LOOP
# =============================================================================

class TFTMainTrainer:
    """
    Main trainer for TFT model with full training loop.

    Handles:
    - Model initialization
    - Loss function setup (0.7 Quantile + 0.3 Sharpe)
    - Optimizer and scheduler setup
    - Training loop with validation
    - Checkpointing based on Sharpe Ratio
    """

    def __init__(
        self,
        config: TFTTrainingConfig,
        device: str = None
    ):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initializing TFTMainTrainer on device: {self.device}")

        # Initialize model
        self.model_config = TFTConfig(
            num_static_features=2,
            num_dynamic_features=config.num_dynamic_features,
            num_known_features=config.num_known_features,
            hidden_size=config.hidden_size,
            lstm_layers=config.lstm_layers,
            attention_heads=config.attention_heads,
            dropout=config.dropout,
            sequence_length=config.sequence_length,
            forecast_horizon=config.forecast_horizon,
            quantiles=config.quantiles
        )

        self.model = SOTAStockPredictor(self.model_config, device=self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Initialize loss function (0.7 Quantile + 0.3 Sharpe)
        # Note: lambda_sharpe is relative weight, we use sharpe_weight / quantile_weight
        lambda_sharpe = config.sharpe_weight / config.quantile_weight
        self.criterion = TradingAwareLoss(
            quantiles=list(config.quantiles),
            lambda_sharpe=lambda_sharpe,
            lambda_sortino=config.sortino_weight,
            lambda_mdd=config.mdd_weight
        )
        logger.info(f"Loss: {config.quantile_weight:.0%} Quantile + {config.sharpe_weight:.0%} Sharpe")

        # Initialize optimizer (AdamW with weight decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        # Scheduler will be initialized when training starts
        self.scheduler = None

        # EMA for model weights
        if config.use_ema:
            self.ema = ExponentialMovingAverage(self.model.model, decay=config.ema_decay)
        else:
            self.ema = None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            model_name='sota_tft'
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_sharpe': [],
            'val_sortino': [],
            'val_dir_acc': [],
            'learning_rates': [],
        }

    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * 10,
                total_steps=num_training_steps,
                pct_start=self.config.onecycle_pct_start,
                div_factor=self.config.onecycle_div_factor
            )
        else:  # cosine
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=15,
                T_mult=2,
                eta_min=1e-7
            )

    def _verify_shapes(self, batch, metadata):
        """Verify batch shapes match model expectations."""
        x_observed, x_known, x_static, y, y_returns = batch

        batch_size = x_observed.shape[0]

        # Assert statements for shape verification
        assert x_static.shape == (batch_size, 2), \
            f"x_static shape {x_static.shape} != expected ({batch_size}, 2)"

        assert x_observed.shape[1] == self.config.sequence_length, \
            f"Sequence length {x_observed.shape[1]} != expected {self.config.sequence_length}"

        assert y.shape == (batch_size, self.config.forecast_horizon), \
            f"Target shape {y.shape} != expected ({batch_size}, {self.config.forecast_horizon})"

        logger.debug(f"Shape verification passed for batch")
        return True

    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_metrics = {}

        for batch_idx, batch in enumerate(train_loader):
            x_observed, x_known, x_static, y, y_returns = batch

            # Move to device
            x_observed = x_observed.to(self.device)
            x_known = x_known.to(self.device)
            x_static = x_static.to(self.device)
            y = y.to(self.device)
            y_returns = y_returns.to(self.device)

            # Pad features if needed
            x_observed, x_known = pad_features_to_model_size(
                x_observed, x_known,
                self.config.num_dynamic_features,
                self.config.num_known_features
            )

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(
                static_features=x_static,
                dynamic_features=x_observed,
                known_features=x_known,
                return_attention=False
            )

            # Get predictions
            predictions = output['quantile_predictions']

            # Compute loss
            # For TFT, target is the median return, expand for quantile loss
            loss, metrics = self.criterion(
                predictions,
                y,  # Target for all horizons
                actual_returns=y_returns  # For Sharpe computation
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()

            # Scheduler step (for OneCycleLR, step per batch)
            if self.config.scheduler_type == 'onecycle' and self.scheduler:
                self.scheduler.step()

            # EMA update
            if self.ema is not None:
                self.ema.update()

            total_loss += loss.item()
            n_batches += 1

            # Log progress
            if batch_idx % 50 == 0:
                logger.debug(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, epoch_metrics

    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()

        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.apply_shadow()

        total_loss = 0.0
        n_batches = 0
        all_sharpe_components = []
        all_predictions = []
        all_targets = []

        for batch in val_loader:
            x_observed, x_known, x_static, y, y_returns = batch

            # Move to device
            x_observed = x_observed.to(self.device)
            x_known = x_known.to(self.device)
            x_static = x_static.to(self.device)
            y = y.to(self.device)
            y_returns = y_returns.to(self.device)

            # Pad features
            x_observed, x_known = pad_features_to_model_size(
                x_observed, x_known,
                self.config.num_dynamic_features,
                self.config.num_known_features
            )

            # Forward pass
            output = self.model(
                static_features=x_static,
                dynamic_features=x_observed,
                known_features=x_known,
                return_attention=False
            )

            predictions = output['quantile_predictions']

            # Compute loss
            loss, metrics = self.criterion(
                predictions, y,
                actual_returns=y_returns
            )

            total_loss += loss.item()
            n_batches += 1

            # Collect predictions for metrics
            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())

            if 'sharpe_sharpe_ratio' in metrics:
                all_sharpe_components.append(metrics['sharpe_sharpe_ratio'])

        # Restore weights
        if self.ema is not None:
            self.ema.restore()

        # Compute metrics
        avg_loss = total_loss / max(n_batches, 1)

        # Aggregate Sharpe
        val_sharpe = np.mean(all_sharpe_components) if all_sharpe_components else 0.0

        # Compute directional accuracy
        all_preds = torch.cat(all_predictions, dim=0)
        all_targs = torch.cat(all_targets, dim=0)

        # Use median prediction for direction
        median_preds = all_preds[:, 0, 1]  # First horizon, median quantile
        dir_accuracy = ((median_preds > 0) == (all_targs[:, 0] > 0)).float().mean().item()

        val_metrics = {
            'val_loss': avg_loss,
            'val_sharpe': val_sharpe,
            'directional_accuracy': dir_accuracy,
        }

        return avg_loss, val_metrics

    def fit(
        self,
        train_loader,
        val_loader,
        metadata: Dict = None
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            metadata: Data metadata for checkpointing

        Returns:
            Training history
        """
        logger.info("=" * 60)
        logger.info("Starting TFT Training")
        logger.info("=" * 60)

        # Verify shapes on first batch
        first_batch = next(iter(train_loader))
        self._verify_shapes(first_batch, metadata)
        logger.info("Shape verification PASSED")

        # Setup scheduler
        num_training_steps = self.config.epochs * len(train_loader)
        self._setup_scheduler(num_training_steps)
        logger.info(f"Scheduler: {self.config.scheduler_type}")
        logger.info(f"Total training steps: {num_training_steps}")

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=1e-5,
            restore_best=True
        )

        best_sharpe = float('-inf')

        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Step scheduler (for cosine, step per epoch)
            if self.config.scheduler_type == 'cosine' and self.scheduler:
                self.scheduler.step()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_sharpe'].append(val_metrics.get('val_sharpe', 0))
            self.history['val_dir_acc'].append(val_metrics.get('directional_accuracy', 0))
            self.history['learning_rates'].append(current_lr)

            # Check for best Sharpe
            current_sharpe = val_metrics.get('val_sharpe', 0)
            is_best = current_sharpe > best_sharpe

            # Log progress
            if epoch % 5 == 0 or is_best:
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"Sharpe: {current_sharpe:.3f} | "
                    f"DirAcc: {val_metrics.get('directional_accuracy', 0):.2%} | "
                    f"LR: {current_lr:.2e}"
                    + (" *BEST*" if is_best else "")
                )

            # Save checkpoint if best Sharpe
            if is_best or epoch % self.config.save_every_n_epochs == 0:
                best_sharpe = max(best_sharpe, current_sharpe)

                ema_state = None
                if self.ema is not None:
                    ema_state = self.ema.shadow.copy()

                self.checkpoint_manager.save_checkpoint(
                    model=self.model.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_sharpe=current_sharpe,
                    config=self.config,
                    metadata=metadata,
                    ema_state=ema_state
                )

            # Early stopping check (on val_loss)
            if early_stopping(val_loss, self.model.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final summary
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Best Validation Sharpe: {best_sharpe:.4f}")
        logger.info(f"Best Validation Loss: {min(self.history['val_loss']):.6f}")
        logger.info(f"Final Directional Accuracy: {self.history['val_dir_acc'][-1]:.2%}")
        logger.info("=" * 60)

        return self.history


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Train TFT Model for Saudi Stock Prediction')

    # Data arguments
    parser.add_argument('--stock_code', type=str, default='2222',
                       help='Stock code to train on (default: 2222 Aramco)')
    parser.add_argument('--period', type=str, default='2y',
                       help='Data period (default: 2y)')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden size (default: 128)')
    parser.add_argument('--lstm_layers', type=int, default=2,
                       help='LSTM layers (default: 2)')
    parser.add_argument('--attention_heads', type=int, default=4,
                       help='Attention heads (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (default: 30)')

    # Loss weights
    parser.add_argument('--quantile_weight', type=float, default=0.7,
                       help='Quantile loss weight (default: 0.7)')
    parser.add_argument('--sharpe_weight', type=float, default=0.3,
                       help='Sharpe loss weight (default: 0.3)')

    # Other
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Checkpoint directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("TFT Training Script - Saudi Stock AI Analyzer")
    logger.info("=" * 70)
    logger.info(f"Stock Code: {args.stock_code}")
    logger.info(f"Period: {args.period}")
    logger.info(f"Device: {args.device or 'auto'}")

    # Load and preprocess data
    logger.info("\n[1/4] Loading data...")
    loader = SaudiStockDataLoader()

    try:
        raw_df = loader.fetch_stock_with_macro(args.stock_code, period=args.period)
    except Exception as e:
        logger.warning(f"Could not load with macro data: {e}")
        raw_df = loader.fetch_stock_data(args.stock_code, period=args.period)

    if raw_df is None or len(raw_df) < 100:
        logger.error("Insufficient data for training")
        return

    logger.info(f"Loaded {len(raw_df)} rows of data")

    # Preprocess
    logger.info("\n[2/4] Preprocessing data...")
    processed_df = preprocess_stock_data(raw_df, include_advanced=True, include_macro=True)
    logger.info(f"Processed data shape: {processed_df.shape}")
    logger.info(f"Features: {len(processed_df.columns)}")

    # Estimate market cap
    market_cap = estimate_market_cap(processed_df, args.stock_code)

    # Create DataLoaders
    logger.info("\n[3/4] Creating DataLoaders...")

    feature_config = TFTFeatureConfig(
        sequence_length=60,
        forecast_horizon=5
    )

    train_loader, val_loader, test_loader, metadata = create_tft_dataloaders(
        df=processed_df,
        stock_code=args.stock_code,
        market_cap=market_cap,
        feature_config=feature_config,
        train_split=0.7,
        val_split=0.15,
        batch_size=args.batch_size
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    logger.info(f"Observed features: {metadata['num_observed_features']}")
    logger.info(f"Known features: {metadata['num_known_features']}")

    # Create training config
    training_config = TFTTrainingConfig(
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        num_dynamic_features=max(40, metadata['num_observed_features']),
        num_known_features=max(5, metadata['num_known_features']),
        sequence_length=feature_config.sequence_length,
        forecast_horizon=feature_config.forecast_horizon,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        quantile_weight=args.quantile_weight,
        sharpe_weight=args.sharpe_weight,
        checkpoint_dir=args.checkpoint_dir or MODEL_SAVE_PATH,
    )

    # Initialize trainer
    logger.info("\n[4/4] Initializing trainer and starting training...")
    trainer = TFTMainTrainer(
        config=training_config,
        device=args.device
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        metadata=metadata
    )

    # Save final training history
    history_path = os.path.join(
        training_config.checkpoint_dir,
        f'training_history_{args.stock_code}.json'
    )
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        json_history = {
            k: [float(v) if isinstance(v, (np.floating, float)) else v for v in vals]
            for k, vals in history.items()
        }
        json.dump(json_history, f, indent=2)

    logger.info(f"\nTraining history saved to: {history_path}")
    logger.info("\nTraining complete!")

    return history


if __name__ == "__main__":
    main()
