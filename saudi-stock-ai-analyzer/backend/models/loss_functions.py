"""
Trading-Aware Loss Functions for SOTA Stock Prediction
=======================================================

This module implements advanced loss functions that combine:
1. Quantile Loss - For uncertainty estimation and probabilistic forecasting
2. Sharpe Loss - For directly optimizing risk-adjusted returns
3. Trading-Aware Loss - Combined objective for financial ML

Copyright (c) 2025 Saudi Market Intelligence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for probabilistic forecasting.

    For a given quantile q:
    L_q(y, ŷ) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)

    This loss penalizes over-predictions and under-predictions asymmetrically
    based on the quantile level, enabling the model to learn prediction intervals.

    Args:
        quantiles: List of quantile levels (e.g., [0.1, 0.5, 0.9])
    """

    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.register_buffer(
            'quantile_weights',
            torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: Shape (batch, seq_len, num_quantiles) or (batch, num_quantiles)
            targets: Shape (batch, seq_len) or (batch,)
            mask: Optional mask for valid positions

        Returns:
            Scalar loss value
        """
        # Ensure targets have quantile dimension for broadcasting
        if predictions.dim() == 3:
            # (batch, seq_len, num_quantiles)
            targets = targets.unsqueeze(-1)  # (batch, seq_len, 1)
        elif predictions.dim() == 2:
            # (batch, num_quantiles)
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)  # (batch, 1)

        # Compute errors
        errors = targets - predictions  # Broadcasting handles quantile dimension

        # Quantile loss: asymmetric penalty
        # For q > 0.5: penalize under-prediction more
        # For q < 0.5: penalize over-prediction more
        quantile_weights = self.quantile_weights.view(1, -1) if predictions.dim() == 2 else \
                          self.quantile_weights.view(1, 1, -1)

        loss = torch.max(
            quantile_weights * errors,
            (quantile_weights - 1) * errors
        )

        # Apply mask if provided
        if mask is not None:
            if mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            return loss.sum() / (mask.sum() * len(self.quantiles) + 1e-8)

        return loss.mean()

    def calibration_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        """
        Compute calibration error for each quantile.

        A well-calibrated model should have q% of observations below
        the q-th quantile prediction.

        Returns:
            Dictionary with expected vs actual coverage for each quantile
        """
        results = {}

        if predictions.dim() == 3:
            targets = targets.unsqueeze(-1)
        elif targets.dim() == 1:
            targets = targets.unsqueeze(-1)

        for i, q in enumerate(self.quantiles):
            pred_q = predictions[..., i:i+1]
            actual_coverage = (targets <= pred_q).float().mean().item()
            expected_coverage = q
            results[f'q{int(q*100)}'] = {
                'expected': expected_coverage,
                'actual': actual_coverage,
                'error': abs(actual_coverage - expected_coverage)
            }

        return results


class SharpeLoss(nn.Module):
    """
    Differentiable Sharpe Ratio Loss for direct optimization of risk-adjusted returns.

    The Sharpe Ratio measures excess return per unit of risk:
    SR = E[R] / Std[R]

    For gradient-based optimization, we use a differentiable approximation:
    Loss = -SR = -mean(returns) / (std(returns) + eps)

    This encourages the model to predict signals that would generate
    high returns with low volatility.

    Args:
        annualization_factor: Factor to annualize Sharpe (252 for daily, 52 for weekly)
        eps: Small constant for numerical stability
        risk_free_rate: Daily risk-free rate (default 0 for simplicity)
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
        eps: float = 1e-8,
        risk_free_rate: float = 0.0
    ):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.eps = eps
        self.risk_free_rate = risk_free_rate

    def forward(
        self,
        predicted_returns: torch.Tensor,
        actual_returns: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute negative Sharpe Ratio as loss.

        Args:
            predicted_returns: Model's return predictions (batch, seq_len) or (batch,)
            actual_returns: Actual market returns (batch, seq_len) or (batch,)
            positions: Optional position sizes based on predictions

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # If positions not provided, use sign of predictions as position
        if positions is None:
            # Soft position sizing using tanh for differentiability
            positions = torch.tanh(predicted_returns)

        # Strategy returns = position * actual_returns
        strategy_returns = positions * actual_returns

        # Flatten for computation
        if strategy_returns.dim() > 1:
            strategy_returns = strategy_returns.reshape(-1)

        # Remove any NaN values
        valid_mask = ~torch.isnan(strategy_returns)
        strategy_returns = strategy_returns[valid_mask]

        if strategy_returns.numel() < 2:
            # Not enough data points
            return torch.tensor(0.0, device=predicted_returns.device), {
                'sharpe_ratio': 0.0,
                'mean_return': 0.0,
                'volatility': 0.0
            }

        # Compute Sharpe components
        excess_returns = strategy_returns - self.risk_free_rate
        mean_return = excess_returns.mean()
        std_return = excess_returns.std()

        # Differentiable Sharpe Ratio
        sharpe_ratio = mean_return / (std_return + self.eps)

        # Annualize
        annualized_sharpe = sharpe_ratio * math.sqrt(self.annualization_factor)

        # Loss is negative Sharpe (we want to maximize Sharpe)
        loss = -annualized_sharpe

        metrics = {
            'sharpe_ratio': annualized_sharpe.item(),
            'mean_return': mean_return.item() * self.annualization_factor,
            'volatility': std_return.item() * math.sqrt(self.annualization_factor),
            'win_rate': (strategy_returns > 0).float().mean().item()
        }

        return loss, metrics


class SortinoLoss(nn.Module):
    """
    Sortino Ratio Loss - focuses on downside risk only.

    Sortino = E[R] / DownsideDeviation

    Unlike Sharpe which penalizes all volatility, Sortino only considers
    negative returns (downside volatility), which is more relevant for trading.
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
        eps: float = 1e-8,
        target_return: float = 0.0
    ):
        super().__init__()
        self.annualization_factor = annualization_factor
        self.eps = eps
        self.target_return = target_return

    def forward(
        self,
        predicted_returns: torch.Tensor,
        actual_returns: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute negative Sortino Ratio as loss."""
        if positions is None:
            positions = torch.tanh(predicted_returns)

        strategy_returns = positions * actual_returns

        if strategy_returns.dim() > 1:
            strategy_returns = strategy_returns.reshape(-1)

        valid_mask = ~torch.isnan(strategy_returns)
        strategy_returns = strategy_returns[valid_mask]

        if strategy_returns.numel() < 2:
            return torch.tensor(0.0, device=predicted_returns.device), {
                'sortino_ratio': 0.0
            }

        # Mean excess return
        mean_return = strategy_returns.mean() - self.target_return

        # Downside deviation (only negative returns)
        downside_returns = torch.clamp(strategy_returns - self.target_return, max=0)
        downside_std = torch.sqrt((downside_returns ** 2).mean() + self.eps)

        # Sortino Ratio
        sortino_ratio = mean_return / downside_std
        annualized_sortino = sortino_ratio * math.sqrt(self.annualization_factor)

        loss = -annualized_sortino

        metrics = {
            'sortino_ratio': annualized_sortino.item(),
            'downside_deviation': downside_std.item() * math.sqrt(self.annualization_factor)
        }

        return loss, metrics


class MaxDrawdownLoss(nn.Module):
    """
    Maximum Drawdown Loss for risk control.

    Drawdown = (Peak - Current) / Peak

    This loss penalizes large drawdowns, encouraging the model to
    avoid catastrophic losses.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        predicted_returns: torch.Tensor,
        actual_returns: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute maximum drawdown as loss."""
        if positions is None:
            positions = torch.tanh(predicted_returns)

        strategy_returns = positions * actual_returns

        # Compute cumulative returns (wealth curve)
        cumulative_returns = torch.cumprod(1 + strategy_returns, dim=-1)

        # Running maximum
        running_max = torch.cummax(cumulative_returns, dim=-1)[0]

        # Drawdowns
        drawdowns = (running_max - cumulative_returns) / (running_max + self.eps)

        # Maximum drawdown
        max_drawdown = drawdowns.max()

        metrics = {
            'max_drawdown': max_drawdown.item(),
            'avg_drawdown': drawdowns.mean().item()
        }

        return max_drawdown, metrics


class TradingAwareLoss(nn.Module):
    """
    Combined Trading-Aware Loss Function.

    This loss combines multiple objectives:
    1. Quantile Loss - For probabilistic prediction accuracy
    2. Sharpe Loss - For risk-adjusted return optimization
    3. Optional Sortino/MaxDrawdown - For additional risk control

    Formula:
    Total_Loss = Quantile_Loss + lambda_sharpe * (-Sharpe_Ratio)
                 + lambda_sortino * (-Sortino_Ratio) + lambda_mdd * MaxDrawdown

    Args:
        quantiles: Quantile levels for probabilistic forecasting
        lambda_sharpe: Weight for Sharpe ratio component
        lambda_sortino: Weight for Sortino ratio component
        lambda_mdd: Weight for max drawdown component
        annualization_factor: For annualizing ratios (252 for daily)
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        lambda_sharpe: float = 0.1,
        lambda_sortino: float = 0.0,
        lambda_mdd: float = 0.0,
        annualization_factor: float = 252.0
    ):
        super().__init__()

        self.quantile_loss = QuantileLoss(quantiles)
        self.sharpe_loss = SharpeLoss(annualization_factor)
        self.sortino_loss = SortinoLoss(annualization_factor) if lambda_sortino > 0 else None
        self.mdd_loss = MaxDrawdownLoss() if lambda_mdd > 0 else None

        self.lambda_sharpe = lambda_sharpe
        self.lambda_sortino = lambda_sortino
        self.lambda_mdd = lambda_mdd
        self.quantiles = quantiles

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        actual_returns: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined trading-aware loss.

        Args:
            predictions: Quantile predictions (batch, seq_len, num_quantiles) or (batch, num_quantiles)
            targets: Actual values (batch, seq_len) or (batch,)
            actual_returns: Actual returns for Sharpe computation (optional)
            mask: Optional mask for valid positions

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        metrics = {}

        # 1. Quantile Loss (always computed)
        q_loss = self.quantile_loss(predictions, targets, mask)
        metrics['quantile_loss'] = q_loss.item()

        total_loss = q_loss

        # 2. Sharpe Loss (if actual returns provided and lambda > 0)
        if actual_returns is not None and self.lambda_sharpe > 0:
            # Use median prediction (q=0.5) as the predicted return
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            predicted_returns = predictions[..., median_idx]

            sharpe_loss, sharpe_metrics = self.sharpe_loss(
                predicted_returns, actual_returns
            )

            total_loss = total_loss + self.lambda_sharpe * sharpe_loss
            metrics.update({f'sharpe_{k}': v for k, v in sharpe_metrics.items()})
            metrics['sharpe_loss_component'] = (self.lambda_sharpe * sharpe_loss).item()

        # 3. Sortino Loss (optional)
        if self.sortino_loss is not None and actual_returns is not None:
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            predicted_returns = predictions[..., median_idx]

            sortino_loss, sortino_metrics = self.sortino_loss(
                predicted_returns, actual_returns
            )

            total_loss = total_loss + self.lambda_sortino * sortino_loss
            metrics.update({f'sortino_{k}': v for k, v in sortino_metrics.items()})

        # 4. Max Drawdown Loss (optional)
        if self.mdd_loss is not None and actual_returns is not None:
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            predicted_returns = predictions[..., median_idx]

            mdd_loss, mdd_metrics = self.mdd_loss(
                predicted_returns, actual_returns
            )

            total_loss = total_loss + self.lambda_mdd * mdd_loss
            metrics.update({f'mdd_{k}': v for k, v in mdd_metrics.items()})

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def get_calibration_report(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        """Generate calibration report for quantile predictions."""
        return self.quantile_loss.calibration_error(predictions, targets)


class DirectionalAccuracyLoss(nn.Module):
    """
    Loss based on direction prediction accuracy.

    In trading, getting the direction right is often more important
    than the exact magnitude. This loss rewards correct direction predictions.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute directional accuracy loss.

        Uses a soft version for differentiability.
        """
        # Get signs (with margin for neutral zone)
        pred_sign = torch.tanh(predictions * 10)  # Soft sign
        target_sign = torch.sign(targets)

        # Directional agreement
        agreement = pred_sign * target_sign

        # Loss: penalize disagreement
        loss = F.relu(-agreement + self.margin).mean()

        # Hard accuracy for metrics
        hard_accuracy = ((predictions > 0) == (targets > 0)).float().mean()

        metrics = {
            'directional_accuracy': hard_accuracy.item(),
            'directional_loss': loss.item()
        }

        return loss, metrics


class ProfitFactorLoss(nn.Module):
    """
    Profit Factor Loss.

    Profit Factor = Gross Profits / Gross Losses

    A PF > 1 indicates profitability. This loss encourages
    the model to generate trading signals with high profit factor.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        predicted_returns: torch.Tensor,
        actual_returns: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute negative log profit factor as loss."""
        if positions is None:
            positions = torch.tanh(predicted_returns)

        strategy_returns = positions * actual_returns

        # Gross profits and losses
        profits = F.relu(strategy_returns)
        losses = F.relu(-strategy_returns)

        gross_profit = profits.sum()
        gross_loss = losses.sum()

        # Profit factor
        profit_factor = gross_profit / (gross_loss + self.eps)

        # Loss: negative log of profit factor (so PF=2 gives lower loss than PF=1)
        loss = -torch.log(profit_factor + self.eps)

        metrics = {
            'profit_factor': profit_factor.item(),
            'gross_profit': gross_profit.item(),
            'gross_loss': gross_loss.item()
        }

        return loss, metrics


# Utility function for computing all trading metrics
def compute_trading_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    actual_returns: torch.Tensor,
    quantiles: List[float] = [0.1, 0.5, 0.9]
) -> dict:
    """
    Compute comprehensive trading metrics.

    Args:
        predictions: Quantile predictions
        targets: Actual target values
        actual_returns: Actual market returns
        quantiles: Quantile levels

    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}

    # Quantile metrics
    q_loss = QuantileLoss(quantiles)
    metrics['quantile_loss'] = q_loss(predictions, targets).item()
    metrics['calibration'] = q_loss.calibration_error(predictions, targets)

    # Get median prediction
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    median_pred = predictions[..., median_idx]

    # Sharpe metrics
    sharpe = SharpeLoss()
    _, sharpe_metrics = sharpe(median_pred, actual_returns)
    metrics.update(sharpe_metrics)

    # Sortino metrics
    sortino = SortinoLoss()
    _, sortino_metrics = sortino(median_pred, actual_returns)
    metrics.update(sortino_metrics)

    # Drawdown metrics
    mdd = MaxDrawdownLoss()
    _, mdd_metrics = mdd(median_pred, actual_returns)
    metrics.update(mdd_metrics)

    # Directional accuracy
    dir_loss = DirectionalAccuracyLoss()
    _, dir_metrics = dir_loss(median_pred, targets)
    metrics.update(dir_metrics)

    # Profit factor
    pf_loss = ProfitFactorLoss()
    _, pf_metrics = pf_loss(median_pred, actual_returns)
    metrics.update(pf_metrics)

    return metrics


if __name__ == "__main__":
    # Test the loss functions
    print("Testing Trading-Aware Loss Functions")
    print("=" * 50)

    # Create sample data
    batch_size = 32
    seq_len = 20
    num_quantiles = 3

    predictions = torch.randn(batch_size, seq_len, num_quantiles)
    # Sort quantiles to be in order
    predictions = torch.sort(predictions, dim=-1)[0]

    targets = torch.randn(batch_size, seq_len)
    actual_returns = torch.randn(batch_size, seq_len) * 0.02  # ~2% daily returns

    # Test TradingAwareLoss
    loss_fn = TradingAwareLoss(
        quantiles=[0.1, 0.5, 0.9],
        lambda_sharpe=0.1,
        lambda_sortino=0.05,
        lambda_mdd=0.01
    )

    loss, metrics = loss_fn(predictions, targets, actual_returns)

    print(f"Total Loss: {loss.item():.4f}")
    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Test calibration
    print("\nCalibration Report:")
    calibration = loss_fn.get_calibration_report(predictions, targets)
    for q, data in calibration.items():
        print(f"  {q}: expected={data['expected']:.2f}, actual={data['actual']:.2f}, error={data['error']:.2f}")

    print("\n" + "=" * 50)
    print("All tests passed!")
