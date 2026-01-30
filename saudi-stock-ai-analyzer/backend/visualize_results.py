#!/usr/bin/env python3
# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
TFT Results Visualization Dashboard
====================================

Interactive visualization of TFT model predictions including:
- Actual vs Predicted prices with uncertainty bands
- Buy/Sell signal markers
- Attention weights heatmap
- Training history plots

Usage:
    python visualize_results.py --checkpoint path/to/checkpoint.pt --stock_code 2222
    python visualize_results.py --use_plotly  # For interactive plots

Author: Claude AI / Abdulrahman Asiri
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_tft import SOTAStockPredictor, TFTConfig
from data.data_loader_tft import (
    create_tft_dataloaders,
    TFTFeatureConfig,
    pad_or_trim_features
)
from data.data_preprocessor import preprocess_stock_data
from data.data_loader import SaudiStockDataLoader
from utils.config import MODEL_SAVE_PATH, SAUDI_STOCKS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT LOADER
# =============================================================================

def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Dict:
    """
    Load a TFT checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Dictionary with model, config, and metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"  Val Sharpe: {checkpoint.get('val_sharpe', 'N/A'):.4f}")
    logger.info(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

    return checkpoint


def create_model_from_checkpoint(checkpoint: Dict, device: str = 'cpu') -> SOTAStockPredictor:
    """Create and load model from checkpoint."""
    config_dict = checkpoint.get('config', {})

    # Create TFTConfig from saved config
    model_config = TFTConfig(
        num_static_features=2,
        num_dynamic_features=config_dict.get('num_dynamic_features', 40),
        num_known_features=config_dict.get('num_known_features', 5),
        hidden_size=config_dict.get('hidden_size', 128),
        lstm_layers=config_dict.get('lstm_layers', 2),
        attention_heads=config_dict.get('attention_heads', 4),
        dropout=config_dict.get('dropout', 0.1),
        sequence_length=config_dict.get('sequence_length', 60),
        forecast_horizon=config_dict.get('forecast_horizon', 5),
        quantiles=tuple(config_dict.get('quantiles', (0.1, 0.5, 0.9)))
    )

    model = SOTAStockPredictor(model_config, device=device)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


# =============================================================================
# PREDICTION GENERATOR
# =============================================================================

@torch.no_grad()
def generate_predictions(
    model: SOTAStockPredictor,
    dataloader,
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Generate predictions with uncertainty bands and attention.

    Returns:
        Dictionary with predictions, actuals, attention weights
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_attention = []
    all_static = []

    config = model.config

    for batch in dataloader:
        x_observed, x_known, x_static, y, y_returns = batch

        # Move to device
        x_observed = x_observed.to(device)
        x_known = x_known.to(device)
        x_static = x_static.to(device)

        # Pad/trim features
        x_observed, x_known = pad_or_trim_features(
            x_observed, x_known,
            target_observed=config.num_dynamic_features,
            target_known=config.num_known_features
        )

        # Forward pass with attention
        output = model(
            static_features=x_static,
            dynamic_features=x_observed,
            known_features=x_known,
            return_attention=True
        )

        all_predictions.append(output['quantile_predictions'].cpu().numpy())
        all_targets.append(y.numpy())

        if 'temporal_attention' in output:
            all_attention.append(output['temporal_attention'].cpu().numpy())

        all_static.append(x_static.cpu().numpy())

    return {
        'predictions': np.concatenate(all_predictions, axis=0),  # (N, horizon, quantiles)
        'targets': np.concatenate(all_targets, axis=0),  # (N, horizon)
        'attention': np.concatenate(all_attention, axis=0) if all_attention else None,
        'static': np.concatenate(all_static, axis=0)
    }


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

def generate_signals(
    predictions: np.ndarray,
    threshold_buy: float = 0.01,
    threshold_sell: float = -0.01,
    confidence_min: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Generate buy/sell signals from predictions.

    Args:
        predictions: Quantile predictions (N, horizon, 3)
        threshold_buy: Minimum predicted return for buy signal
        threshold_sell: Maximum predicted return for sell signal
        confidence_min: Minimum confidence (1 - uncertainty) for signal

    Returns:
        Dictionary with signal arrays
    """
    # Extract quantiles
    q10 = predictions[:, 0, 0]  # 10th percentile, first horizon
    q50 = predictions[:, 0, 1]  # Median, first horizon
    q90 = predictions[:, 0, 2]  # 90th percentile, first horizon

    # Uncertainty = width of interval
    uncertainty = q90 - q10
    confidence = 1.0 / (1.0 + np.abs(uncertainty))

    # Normalize confidence to 0-1 range
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)

    # Generate signals
    buy_signal = (q50 > threshold_buy) & (confidence >= confidence_min)
    sell_signal = (q50 < threshold_sell) & (confidence >= confidence_min)

    return {
        'buy': buy_signal,
        'sell': sell_signal,
        'median': q50,
        'low': q10,
        'high': q90,
        'confidence': confidence
    }


# =============================================================================
# MATPLOTLIB VISUALIZATIONS
# =============================================================================

def plot_predictions_matplotlib(
    dates: np.ndarray,
    actual_prices: np.ndarray,
    predictions: Dict,
    signals: Dict,
    stock_code: str,
    save_path: Optional[str] = None
):
    """
    Plot predictions with uncertainty bands using matplotlib.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[3, 1, 1])
    fig.suptitle(f'TFT Predictions for {stock_code} ({SAUDI_STOCKS.get(stock_code, {}).get("name", "Unknown")})',
                 fontsize=14, fontweight='bold')

    # === Panel 1: Price with Uncertainty Bands ===
    ax1 = axes[0]

    # Convert predictions (returns) to prices
    base_price = actual_prices[0]
    pred_median = predictions['targets'][:, 0]  # Actual returns for comparison

    # Plot actual prices
    ax1.plot(dates, actual_prices, 'b-', linewidth=1.5, label='Actual Price', alpha=0.8)

    # Create uncertainty band in price space
    # Note: We're showing the prediction uncertainty as a percentage band
    uncertainty_pct = (signals['high'] - signals['low']) / 2
    price_upper = actual_prices * (1 + uncertainty_pct)
    price_lower = actual_prices * (1 - uncertainty_pct)

    ax1.fill_between(dates, price_lower, price_upper,
                     alpha=0.3, color='orange', label='Uncertainty Band (10%-90%)')

    # Plot buy/sell signals
    buy_dates = dates[signals['buy']]
    buy_prices = actual_prices[signals['buy']]
    sell_dates = dates[signals['sell']]
    sell_prices = actual_prices[signals['sell']]

    ax1.scatter(buy_dates, buy_prices, marker='^', s=100, c='green',
                label=f'Buy Signal ({len(buy_dates)})', zorder=5, edgecolors='darkgreen')
    ax1.scatter(sell_dates, sell_prices, marker='v', s=100, c='red',
                label=f'Sell Signal ({len(sell_dates)})', zorder=5, edgecolors='darkred')

    ax1.set_ylabel('Price (SAR)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price Prediction with Uncertainty', fontsize=12)

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # === Panel 2: Predicted Returns with Confidence ===
    ax2 = axes[1]

    # Plot predicted median returns
    ax2.fill_between(dates, signals['low'] * 100, signals['high'] * 100,
                     alpha=0.3, color='purple', label='Prediction Interval')
    ax2.plot(dates, signals['median'] * 100, 'purple', linewidth=1.5, label='Predicted Return')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Highlight buy/sell regions
    for i, (buy, sell) in enumerate(zip(signals['buy'], signals['sell'])):
        if buy:
            ax2.axvspan(dates[i], dates[min(i+1, len(dates)-1)], alpha=0.2, color='green')
        if sell:
            ax2.axvspan(dates[i], dates[min(i+1, len(dates)-1)], alpha=0.2, color='red')

    ax2.set_ylabel('Predicted Return (%)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Predicted Returns with Uncertainty', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # === Panel 3: Confidence Score ===
    ax3 = axes[2]

    # Confidence as bar chart
    colors = ['green' if c > 0.7 else 'orange' if c > 0.4 else 'red'
              for c in signals['confidence']]
    ax3.bar(dates, signals['confidence'] * 100, color=colors, alpha=0.7, width=1)
    ax3.axhline(y=50, color='orange', linestyle='--', linewidth=1, label='Min Confidence')
    ax3.set_ylabel('Confidence (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Model Confidence', fontsize=12)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to: {save_path}")

    plt.show()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    sample_idx: int = -1,
    save_path: Optional[str] = None
):
    """
    Plot attention weights heatmap.

    Args:
        attention_weights: Attention matrix (N, seq_len, seq_len)
        sample_idx: Which sample to visualize (-1 for last)
    """
    import matplotlib.pyplot as plt

    attn = attention_weights[sample_idx]  # (seq_len, seq_len)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(attn, cmap='viridis', aspect='auto')

    ax.set_xlabel('Key Position (Past Days)', fontsize=11)
    ax.set_ylabel('Query Position (Current Day)', fontsize=11)
    ax.set_title('Temporal Attention Weights\n(Brighter = More Attention)', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=10)

    # Add annotations for top attention
    seq_len = attn.shape[0]
    if seq_len <= 20:  # Only annotate for small sequences
        for i in range(seq_len):
            for j in range(seq_len):
                if attn[i, j] > 0.1:
                    ax.text(j, i, f'{attn[i, j]:.2f}',
                           ha='center', va='center', fontsize=6, color='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved attention heatmap to: {save_path}")

    plt.show()


def plot_training_history(
    history_path: str,
    save_path: Optional[str] = None
):
    """Plot training history from JSON file."""
    import matplotlib.pyplot as plt
    import json

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TFT Training History', fontsize=14, fontweight='bold')

    # Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sharpe Ratio
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_sharpe'], 'g-', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.fill_between(epochs, 0, history['val_sharpe'],
                     where=[s > 0 for s in history['val_sharpe']],
                     alpha=0.3, color='green')
    ax2.fill_between(epochs, 0, history['val_sharpe'],
                     where=[s <= 0 for s in history['val_sharpe']],
                     alpha=0.3, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Validation Sharpe Ratio')
    ax2.grid(True, alpha=0.3)

    # Directional Accuracy
    ax3 = axes[1, 0]
    ax3.plot(epochs, [d * 100 for d in history['val_dir_acc']], 'm-', linewidth=2)
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Random (50%)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Directional Accuracy')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Learning Rate
    ax4 = axes[1, 1]
    ax4.semilogy(epochs, history['learning_rates'], 'c-', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate (log scale)')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training history plot to: {save_path}")

    plt.show()


# =============================================================================
# PLOTLY INTERACTIVE VISUALIZATIONS
# =============================================================================

def plot_predictions_plotly(
    dates: np.ndarray,
    actual_prices: np.ndarray,
    predictions: Dict,
    signals: Dict,
    stock_code: str,
    save_path: Optional[str] = None
):
    """
    Create interactive plotly visualization.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("Plotly not installed. Install with: pip install plotly")
        logger.warning("Falling back to matplotlib...")
        return plot_predictions_matplotlib(dates, actual_prices, predictions, signals, stock_code, save_path)

    stock_name = SAUDI_STOCKS.get(stock_code, {}).get('name', 'Unknown')

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            'Price with Uncertainty Band',
            'Predicted Returns',
            'Model Confidence'
        ),
        row_heights=[0.5, 0.25, 0.25]
    )

    # === Row 1: Price with Uncertainty ===
    # Actual price
    fig.add_trace(
        go.Scatter(
            x=dates, y=actual_prices,
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Uncertainty band
    uncertainty_pct = (signals['high'] - signals['low']) / 2
    price_upper = actual_prices * (1 + uncertainty_pct)
    price_lower = actual_prices * (1 - uncertainty_pct)

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([price_upper, price_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.3)',
            line=dict(color='rgba(255,165,0,0)'),
            name='Uncertainty Band (10%-90%)',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # Buy signals
    buy_mask = signals['buy']
    if buy_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[buy_mask],
                y=actual_prices[buy_mask],
                mode='markers',
                name=f'Buy Signal ({buy_mask.sum()})',
                marker=dict(symbol='triangle-up', size=12, color='green',
                           line=dict(width=1, color='darkgreen'))
            ),
            row=1, col=1
        )

    # Sell signals
    sell_mask = signals['sell']
    if sell_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[sell_mask],
                y=actual_prices[sell_mask],
                mode='markers',
                name=f'Sell Signal ({sell_mask.sum()})',
                marker=dict(symbol='triangle-down', size=12, color='red',
                           line=dict(width=1, color='darkred'))
            ),
            row=1, col=1
        )

    # === Row 2: Predicted Returns ===
    # Uncertainty band for returns
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([signals['high'] * 100, (signals['low'] * 100)[::-1]]),
            fill='toself',
            fillcolor='rgba(128, 0, 128, 0.3)',
            line=dict(color='rgba(128,0,128,0)'),
            name='Return Interval',
            hoverinfo='skip',
            showlegend=False
        ),
        row=2, col=1
    )

    # Median prediction
    fig.add_trace(
        go.Scatter(
            x=dates, y=signals['median'] * 100,
            mode='lines',
            name='Predicted Return',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # === Row 3: Confidence ===
    colors = ['green' if c > 0.7 else 'orange' if c > 0.4 else 'red'
              for c in signals['confidence']]

    fig.add_trace(
        go.Bar(
            x=dates, y=signals['confidence'] * 100,
            name='Confidence',
            marker_color=colors,
            opacity=0.7
        ),
        row=3, col=1
    )

    fig.add_hline(y=50, line_dash="dash", line_color="orange", row=3, col=1)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'TFT Predictions: {stock_code} - {stock_name}',
            font=dict(size=16)
        ),
        height=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )

    # Update axes
    fig.update_yaxes(title_text="Price (SAR)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Confidence (%)", range=[0, 100], row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved interactive plot to: {save_path}")

    fig.show()


def plot_attention_plotly(
    attention_weights: np.ndarray,
    sample_idx: int = -1,
    save_path: Optional[str] = None
):
    """Interactive attention heatmap with plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("Plotly not installed. Falling back to matplotlib...")
        return plot_attention_heatmap(attention_weights, sample_idx, save_path)

    attn = attention_weights[sample_idx]
    seq_len = attn.shape[0]

    fig = go.Figure(data=go.Heatmap(
        z=attn,
        colorscale='Viridis',
        colorbar=dict(title='Attention Weight'),
        hovertemplate='Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title='Temporal Attention Weights<br><sub>Which past days influenced the prediction?</sub>',
        xaxis_title='Key Position (Past Days)',
        yaxis_title='Query Position (Current Day)',
        height=600,
        width=800
    )

    # Reverse y-axis so most recent is at top
    fig.update_yaxes(autorange='reversed')

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved attention heatmap to: {save_path}")

    fig.show()


# =============================================================================
# MAIN VISUALIZATION PIPELINE
# =============================================================================

def run_visualization(
    checkpoint_path: str,
    stock_code: str = '2222',
    period: str = '1y',
    use_plotly: bool = True,
    output_dir: Optional[str] = None
):
    """
    Main visualization pipeline.

    Args:
        checkpoint_path: Path to model checkpoint
        stock_code: Stock code to visualize
        period: Data period for visualization
        use_plotly: Use interactive plotly (vs matplotlib)
        output_dir: Directory to save plots
    """
    logger.info("=" * 60)
    logger.info("TFT Results Visualization")
    logger.info("=" * 60)

    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(checkpoint_path), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    logger.info(f"\n[1/5] Loading checkpoint...")
    checkpoint = load_checkpoint(checkpoint_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model_from_checkpoint(checkpoint, device)

    # Load and preprocess data
    logger.info(f"\n[2/5] Loading data for {stock_code}...")
    loader = SaudiStockDataLoader()

    try:
        df = loader.fetch_stock_with_macro(stock_code, period=period)
    except Exception:
        df = loader.fetch_stock_data(stock_code, period=period)

    if df is None or len(df) < 100:
        raise ValueError(f"Insufficient data for {stock_code}")

    processed_df = preprocess_stock_data(df, include_advanced=True, include_macro=True)
    logger.info(f"  Data: {len(processed_df)} rows")

    # Create test dataloader
    logger.info(f"\n[3/5] Creating dataloader...")
    feature_config = TFTFeatureConfig(
        sequence_length=checkpoint['config'].get('sequence_length', 60),
        forecast_horizon=checkpoint['config'].get('forecast_horizon', 5)
    )

    _, _, test_loader, metadata = create_tft_dataloaders(
        df=processed_df,
        stock_code=stock_code,
        feature_config=feature_config,
        train_split=0.8,
        val_split=0.1,
        batch_size=32
    )

    # Generate predictions
    logger.info(f"\n[4/5] Generating predictions...")
    results = generate_predictions(model, test_loader, device)

    # Generate signals
    signals = generate_signals(
        results['predictions'],
        threshold_buy=0.005,
        threshold_sell=-0.005,
        confidence_min=0.4
    )

    logger.info(f"  Buy signals: {signals['buy'].sum()}")
    logger.info(f"  Sell signals: {signals['sell'].sum()}")

    # Prepare dates and prices for plotting
    # Use the last N rows of processed_df matching our predictions
    n_samples = len(results['predictions'])
    plot_df = processed_df.tail(n_samples + feature_config.sequence_length).iloc[feature_config.sequence_length:]
    plot_df = plot_df.head(n_samples)

    dates = plot_df.index.values
    actual_prices = plot_df['Close'].values

    # Plot
    logger.info(f"\n[5/5] Creating visualizations...")

    if use_plotly:
        # Interactive plots
        plot_predictions_plotly(
            dates, actual_prices, results, signals, stock_code,
            save_path=os.path.join(output_dir, f'{stock_code}_predictions.html')
        )

        if results['attention'] is not None:
            plot_attention_plotly(
                results['attention'],
                sample_idx=-1,
                save_path=os.path.join(output_dir, f'{stock_code}_attention.html')
            )
    else:
        # Static plots
        plot_predictions_matplotlib(
            dates, actual_prices, results, signals, stock_code,
            save_path=os.path.join(output_dir, f'{stock_code}_predictions.png')
        )

        if results['attention'] is not None:
            plot_attention_heatmap(
                results['attention'],
                sample_idx=-1,
                save_path=os.path.join(output_dir, f'{stock_code}_attention.png')
            )

    # Plot training history if available
    history_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f'training_history_{stock_code}.json'
    )
    if os.path.exists(history_path):
        plot_training_history(
            history_path,
            save_path=os.path.join(output_dir, f'{stock_code}_training_history.png')
        )

    logger.info("\n" + "=" * 60)
    logger.info(f"Visualizations saved to: {output_dir}")
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize TFT Prediction Results')

    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (default: best_sharpe in MODEL_SAVE_PATH)')
    parser.add_argument('--stock_code', type=str, default='2222',
                       help='Stock code to visualize (default: 2222)')
    parser.add_argument('--period', type=str, default='1y',
                       help='Data period (default: 1y)')
    parser.add_argument('--use_plotly', action='store_true',
                       help='Use interactive plotly (default: matplotlib)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')

    args = parser.parse_args()

    # Find checkpoint if not specified
    if args.checkpoint is None:
        default_checkpoint = os.path.join(MODEL_SAVE_PATH, 'sota_tft_best_sharpe.pt')
        if os.path.exists(default_checkpoint):
            args.checkpoint = default_checkpoint
        else:
            # Look for any checkpoint
            checkpoints = [f for f in os.listdir(MODEL_SAVE_PATH) if f.endswith('.pt')]
            if checkpoints:
                args.checkpoint = os.path.join(MODEL_SAVE_PATH, checkpoints[0])
            else:
                raise FileNotFoundError(
                    f"No checkpoint found. Please train a model first or specify --checkpoint"
                )

    run_visualization(
        checkpoint_path=args.checkpoint,
        stock_code=args.stock_code,
        period=args.period,
        use_plotly=args.use_plotly,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
