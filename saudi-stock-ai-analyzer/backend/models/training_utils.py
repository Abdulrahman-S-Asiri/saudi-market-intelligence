# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
Advanced Training Utilities for High-Accuracy LSTM Model

This module provides training enhancements including:
- Cosine Annealing Learning Rate with Warm Restarts
- Label Smoothing
- Mixup Data Augmentation
- Ensemble Training
- Early Stopping with Model Checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warm Restarts Learning Rate Scheduler.

    The learning rate follows a cosine schedule and restarts periodically.
    This helps escape local minima and find better optima.

    Reference:
        Loshchilov & Hutter (2016). SGDR: Stochastic Gradient Descent
        with Warm Restarts.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Wrapped optimizer
            T_0: Initial restart period
            T_mult: Period multiplier after each restart
            eta_min: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class OneCycleLR:
    """
    One-Cycle Learning Rate Policy.

    Implements the 1cycle policy: increases LR from low to high,
    then decreases to very low. Often leads to faster convergence.

    Reference:
        Smith & Topin (2019). Super-Convergence: Very Fast Training
        of Neural Networks Using Large Learning Rates.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / (div_factor * final_div_factor)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        pct = self.step_count / self.total_steps

        if pct < self.pct_start:
            # Warm up phase
            scale = pct / self.pct_start
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * scale
        else:
            # Annealing phase
            pct_post = (pct - self.pct_start) / (1 - self.pct_start)
            if self.anneal_strategy == 'cos':
                lr = self.final_lr + (self.max_lr - self.final_lr) * (1 + np.cos(np.pi * pct_post)) / 2
            else:
                lr = self.max_lr - (self.max_lr - self.final_lr) * pct_post

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class LabelSmoothing(nn.Module):
    """
    Label Smoothing for Regression.

    Adds noise to targets during training to prevent overconfidence
    and improve generalization.
    """

    def __init__(self, smoothing: float = 0.1, noise_type: str = 'gaussian'):
        """
        Initialize label smoothing.

        Args:
            smoothing: Amount of smoothing (0.0 to 1.0)
            noise_type: Type of noise ('gaussian' or 'uniform')
        """
        super().__init__()
        self.smoothing = smoothing
        self.noise_type = noise_type

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        if self.training and self.smoothing > 0:
            if self.noise_type == 'gaussian':
                noise = torch.randn_like(targets) * self.smoothing
            else:
                noise = (torch.rand_like(targets) - 0.5) * 2 * self.smoothing

            return targets + noise
        return targets


class MixupAugmentation:
    """
    Mixup Data Augmentation.

    Creates virtual training samples by linear interpolation
    between pairs of samples.

    Reference:
        Zhang et al. (2018). mixup: Beyond Empirical Risk Minimization.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initialize Mixup.

        Args:
            alpha: Beta distribution parameter (higher = more mixing)
        """
        self.alpha = alpha

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation.

        Args:
            x: Input tensor (batch_size, seq_len, features)
            y: Target tensor (batch_size,)

        Returns:
            Mixed inputs, mixed targets, lambda value
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y, lam


class CutMix:
    """
    CutMix Data Augmentation for Time Series.

    Cuts segments from one sample and pastes into another.

    Reference:
        Yun et al. (2019). CutMix: Regularization Strategy
    """

    def __init__(self, alpha: float = 1.0, probability: float = 0.5):
        self.alpha = alpha
        self.probability = probability

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if np.random.random() > self.probability:
            return x, y, 1.0

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, seq_len, features = x.shape

        # Determine cut region
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len + 1)
        cut_end = cut_start + cut_len

        # Shuffle indices
        index = torch.randperm(batch_size, device=x.device)

        # Apply CutMix
        mixed_x = x.clone()
        mixed_x[:, cut_start:cut_end, :] = x[index, cut_start:cut_end, :]

        # Adjust lambda based on actual cut ratio
        lam_adjusted = 1 - (cut_len / seq_len)
        mixed_y = lam_adjusted * y + (1 - lam_adjusted) * y[index]

        return mixed_x, mixed_y, lam_adjusted


class EarlyStopping:
    """
    Early Stopping with Model Checkpointing.

    Stops training when validation loss stops improving and saves
    the best model checkpoint.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        restore_best: bool = True,
        save_path: str = None
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best: Whether to restore best weights on stop
            save_path: Path to save best model checkpoint
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.save_path = save_path

        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        self.should_stop = False

    def __call__(
        self,
        val_loss: float,
        model: nn.Module
    ) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            model: Model to checkpoint

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

            if self.save_path:
                torch.save(self.best_model_state, self.save_path)

            return False
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

                if self.restore_best and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)

                return True

        return False

    def restore(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class GradientAccumulator:
    """
    Gradient Accumulation for Larger Effective Batch Sizes.

    Useful when GPU memory is limited but larger batch sizes improve training.
    """

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_step(self) -> bool:
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0

    def reset(self):
        self.current_step = 0


class ExponentialMovingAverage:
    """
    Exponential Moving Average of Model Weights.

    Maintains a shadow copy of model weights that is updated slowly,
    often leading to better generalization.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class AdvancedTrainer:
    """
    Advanced Training Loop with All Enhancements.

    Combines all training utilities for high-performance model training.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
        device: str = None,
        use_mixup: bool = True,
        use_label_smoothing: bool = True,
        use_ema: bool = True,
        use_gradient_accumulation: bool = False,
        mixup_alpha: float = 0.2,
        label_smoothing: float = 0.1,
        ema_decay: float = 0.999,
        accumulation_steps: int = 4,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize advanced trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer (default: AdamW)
            criterion: Loss function
            device: Device to train on
            use_mixup: Whether to use Mixup augmentation
            use_label_smoothing: Whether to use label smoothing
            use_ema: Whether to use EMA of weights
            use_gradient_accumulation: Whether to accumulate gradients
            mixup_alpha: Mixup alpha parameter
            label_smoothing: Label smoothing factor
            ema_decay: EMA decay rate
            accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=0.0005,
            weight_decay=0.01
        )

        self.criterion = criterion or nn.MSELoss()
        self.max_grad_norm = max_grad_norm

        # Augmentation and regularization
        self.mixup = MixupAugmentation(alpha=mixup_alpha) if use_mixup else None
        self.label_smoothing = LabelSmoothing(smoothing=label_smoothing) if use_label_smoothing else None
        self.ema = ExponentialMovingAverage(model, decay=ema_decay) if use_ema else None
        self.gradient_accumulator = GradientAccumulator(accumulation_steps) if use_gradient_accumulation else None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        scheduler=None
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            scheduler: Optional learning rate scheduler

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device).float()
            y = y.to(self.device).float()

            # Apply Mixup
            if self.mixup is not None:
                x, y, _ = self.mixup(x, y)

            # Apply Label Smoothing
            if self.label_smoothing is not None:
                y = self.label_smoothing(y)

            # Forward pass
            output = self.model(x)
            prediction = output['prediction'].squeeze()
            loss = self.criterion(prediction, y)

            # Add uncertainty regularization if available
            if 'uncertainty' in output:
                uncertainty_reg = output['uncertainty'].mean() * 0.01
                loss = loss + uncertainty_reg

            # Backward pass
            if self.gradient_accumulator is not None:
                loss = loss / self.gradient_accumulator.accumulation_steps
                loss.backward()

                if self.gradient_accumulator.should_step():
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.ema is not None:
                        self.ema.update()
            else:
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()

                if self.ema is not None:
                    self.ema.update()

            total_loss += loss.item()
            n_batches += 1

            if scheduler is not None:
                scheduler.step()

        return total_loss / n_batches

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()

        # Use EMA weights for validation if available
        if self.ema is not None:
            self.ema.apply_shadow()

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float()

                output = self.model(x)
                prediction = output['prediction'].squeeze()
                loss = self.criterion(prediction, y)

                total_loss += loss.item()
                n_batches += 1

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        return total_loss / n_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        patience: int = 20,
        scheduler_type: str = 'cosine',
        verbose: bool = True,
        save_dir: str = None
    ) -> Dict:
        """
        Full training loop with all enhancements.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum epochs
            patience: Early stopping patience
            scheduler_type: LR scheduler type ('cosine', 'onecycle', 'none')
            verbose: Whether to print progress
            save_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        # Setup scheduler
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
            step_per_batch = False
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]['lr'] * 10,
                total_steps=epochs * len(train_loader)
            )
            step_per_batch = True
        else:
            scheduler = None
            step_per_batch = False

        # Setup early stopping
        save_path = os.path.join(save_dir, 'best_model.pt') if save_dir else None
        early_stopping = EarlyStopping(
            patience=patience,
            restore_best=True,
            save_path=save_path
        )

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            if step_per_batch:
                train_loss = self.train_epoch(train_loader, scheduler)
            else:
                train_loss = self.train_epoch(train_loader)
                if scheduler is not None:
                    scheduler.step()

            # Validate
            val_loss = self.validate(val_loader)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if verbose and epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, "
                    f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # Early stopping check
            if early_stopping(val_loss, self.model):
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break

        return self.history


class EnsembleTrainer:
    """
    Train an ensemble of models with different random seeds.
    """

    def __init__(
        self,
        model_class,
        model_kwargs: Dict,
        n_models: int = 5,
        base_seed: int = 42
    ):
        """
        Initialize ensemble trainer.

        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model
            n_models: Number of models in ensemble
            base_seed: Base random seed
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.n_models = n_models
        self.base_seed = base_seed
        self.models = []
        self.trainers = []

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **train_kwargs
    ) -> List[Dict]:
        """
        Train all ensemble models.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            **train_kwargs: Training arguments

        Returns:
            List of training histories
        """
        histories = []

        for i in range(self.n_models):
            logger.info(f"Training model {i+1}/{self.n_models}")

            # Set seed for reproducibility
            seed = self.base_seed + i * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create model
            model = self.model_class(**self.model_kwargs)

            # Create trainer
            trainer = AdvancedTrainer(model)
            self.trainers.append(trainer)

            # Train
            history = trainer.fit(train_loader, val_loader, **train_kwargs)
            histories.append(history)

            self.models.append(model)

        return histories

    def predict(self, x: np.ndarray, aggregation: str = 'mean') -> Dict:
        """
        Make ensemble predictions.

        Args:
            x: Input data
            aggregation: Aggregation method ('mean' or 'median')

        Returns:
            Aggregated predictions with uncertainty
        """
        device = self.trainers[0].device if self.trainers else 'cpu'

        predictions = []
        confidences = []

        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            for model in self.models:
                model.eval()
                output = model(x_tensor)
                predictions.append(output['prediction'].cpu().numpy())
                if 'confidence' in output:
                    confidences.append(output['confidence'].cpu().numpy())

        predictions = np.array(predictions)

        if aggregation == 'mean':
            ensemble_pred = predictions.mean(axis=0)
        else:
            ensemble_pred = np.median(predictions, axis=0)

        # Ensemble uncertainty from disagreement
        ensemble_std = predictions.std(axis=0)

        result = {
            'prediction': ensemble_pred,
            'std': ensemble_std,
            'all_predictions': predictions
        }

        if confidences:
            confidences = np.array(confidences)
            result['confidence'] = confidences.mean(axis=0) * np.exp(-ensemble_std)

        return result


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders from numpy arrays.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
        shuffle: Whether to shuffle training data

    Returns:
        train_loader, val_loader
    """
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Training Utilities...")
    print("=" * 60)

    # Test Mixup
    x = torch.randn(32, 60, 35)
    y = torch.randn(32)
    mixup = MixupAugmentation(alpha=0.2)
    mixed_x, mixed_y, lam = mixup(x, y)
    print(f"Mixup: lambda={lam:.3f}")

    # Test Label Smoothing
    ls = LabelSmoothing(smoothing=0.1)
    ls.train()
    smoothed_y = ls(y)
    print(f"Label smoothing: original range [{y.min():.3f}, {y.max():.3f}], "
          f"smoothed range [{smoothed_y.min():.3f}, {smoothed_y.max():.3f}]")

    # Test Cosine Annealing
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    lrs = []
    for _ in range(50):
        lrs.append(scheduler.get_lr()[0])
        scheduler.step()

    print(f"Cosine annealing LR range: [{min(lrs):.6f}, {max(lrs):.6f}]")

    # Test Early Stopping
    es = EarlyStopping(patience=3)
    losses = [0.5, 0.4, 0.3, 0.35, 0.36, 0.37, 0.38]
    for i, loss in enumerate(losses):
        should_stop = es(loss, model)
        print(f"  Epoch {i}: loss={loss}, should_stop={should_stop}")
        if should_stop:
            break

    print("\nTraining Utilities Test: OK")
