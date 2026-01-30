# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
TFT Data Loader and Dataset
============================

Custom PyTorch Dataset and DataLoader utilities for the Temporal Fusion Transformer.

This module handles:
- Feature splitting into static, known, and observed categories
- Proper train/val/test scaling (fit on train only to avoid data leakage)
- Sequence creation with correct shapes for TFT input
- Calendar feature extraction for known future inputs

Input Shape Requirements for SOTAStockPredictor:
- static_features: (batch, 2) - [sector_id, market_cap_group]
- dynamic_features: (batch, seq_len, num_dynamic_features)
- known_features: (batch, seq_len, num_known_features) - optional
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from dataclasses import dataclass
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    ADVANCED_LSTM_FEATURES,
    ADVANCED_LSTM_CONFIG,
    SECTORS,
    SAUDI_STOCKS
)

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

@dataclass
class TFTFeatureConfig:
    """
    Configuration for TFT feature categories.

    TFT requires splitting features into three categories:
    1. Static (time-invariant): Sector, Market Cap Group
    2. Known (future-available): Calendar features, scheduled events
    3. Observed (past-only): Price, Volume, Indicators
    """
    # Static features (do not change over time for a given stock)
    static_features: List[str] = None

    # Known features (calendar-based, known in advance)
    known_features: List[str] = None

    # Observed/Dynamic features (only available up to current time)
    observed_features: List[str] = None

    # Target column(s)
    target_column: str = 'Close'
    target_returns: bool = True  # Use returns instead of raw prices

    # Sequence parameters
    sequence_length: int = 60
    forecast_horizon: int = 5

    def __post_init__(self):
        """Set default feature lists if not provided."""
        if self.static_features is None:
            self.static_features = ['sector_id', 'market_cap_group']

        if self.known_features is None:
            # Calendar features that are known in advance
            self.known_features = [
                'day_of_week',
                'day_of_month',
                'month',
                'quarter',
                'is_month_end',
            ]

        if self.observed_features is None:
            # Use the advanced LSTM features as base
            self.observed_features = [
                # Price-based
                'Close', 'High', 'Low', 'Volume',
                # Moving averages
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                # Momentum
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                # Volatility
                'ATR', 'BB_Width', 'Volatility', 'HV_20',
                # Volume
                'OBV', 'Volume_MA', 'Volume_Ratio', 'Relative_Volume',
                # Oscillators
                'Stoch_K', 'Williams_R', 'ADX', 'CCI', 'MFI', 'Ultimate_Osc',
                # Microstructure
                'Volume_Momentum', 'Price_Volume_Corr', 'Amihud_Illiquidity',
                # Channels
                'Keltner_Position', 'Donchian_Position',
                # Pattern
                'Gap_Percentage', 'Body_Ratio', 'Dist_From_20d_High', 'Dist_From_20d_Low',
                # Statistical
                'Price_Zscore', 'Price_Percentile', 'Return_Skewness', 'Vol_Regime',
                # Trend
                'ROC', 'Momentum', 'Daily_Return', 'Choppiness', 'TRIX',
                # Cross-Asset (if available)
                'Oil_Correlation', 'Market_Trend', 'Stock_Market_Beta',
            ]


# =============================================================================
# SECTOR AND MARKET CAP MAPPING
# =============================================================================

# Map sector names to integer IDs (0-19 for TASI)
SECTOR_TO_ID = {
    'Energy': 0,
    'Materials': 1,
    'Banks': 2,
    'Diversified Financials': 3,
    'Insurance': 4,
    'Telecommunication Services': 5,
    'Utilities': 6,
    'Real Estate': 7,
    'REITs': 8,
    'Consumer Services': 9,
    'Food & Staples Retailing': 10,
    'Food & Beverages': 11,
    'Health Care': 12,
    'Pharma': 13,
    'Capital Goods': 14,
    'Transportation': 15,
    'Software & Services': 16,
    'Consumer Durables': 17,
    'Unknown': 18,
}

# Market cap groups based on Saudi market thresholds (SAR billions)
MARKET_CAP_GROUPS = {
    'Large': 0,    # > 50B SAR
    'Mid': 1,      # 10B - 50B SAR
    'Small': 2,    # 2B - 10B SAR
    'Micro': 3,    # < 2B SAR
}


def get_sector_id(stock_code: str) -> int:
    """Get sector ID for a stock code."""
    if stock_code in SAUDI_STOCKS:
        sector = SAUDI_STOCKS[stock_code].get('sector', 'Unknown')
        return SECTOR_TO_ID.get(sector, SECTOR_TO_ID['Unknown'])
    return SECTOR_TO_ID['Unknown']


def get_market_cap_group(market_cap_sar: float) -> int:
    """
    Get market cap group ID based on market cap in SAR.

    Args:
        market_cap_sar: Market cap in SAR (not billions)

    Returns:
        Market cap group ID (0-3)
    """
    market_cap_billions = market_cap_sar / 1e9

    if market_cap_billions > 50:
        return MARKET_CAP_GROUPS['Large']
    elif market_cap_billions > 10:
        return MARKET_CAP_GROUPS['Mid']
    elif market_cap_billions > 2:
        return MARKET_CAP_GROUPS['Small']
    else:
        return MARKET_CAP_GROUPS['Micro']


# =============================================================================
# TFT DATASET
# =============================================================================

class TFTDataset(Dataset):
    """
    PyTorch Dataset for Temporal Fusion Transformer.

    Handles proper feature splitting into:
    - x_static: Static covariates (sector, market_cap)
    - x_known: Known future inputs (calendar features)
    - x_observed: Observed past inputs (price, volume, indicators)
    - y: Target values (returns or prices)

    All features are properly scaled using scalers fit only on training data.

    Args:
        df: DataFrame with all features
        feature_config: TFTFeatureConfig with feature definitions
        stock_code: Stock ticker code for static features
        market_cap: Market cap in SAR for static features
        scaler_observed: Optional pre-fitted scaler for observed features
        scaler_known: Optional pre-fitted scaler for known features
        is_training: Whether this is training data (for fitting scalers)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_config: TFTFeatureConfig,
        stock_code: str = '2222',
        market_cap: float = 1e12,  # Default: 1 trillion SAR (Aramco-scale)
        scaler_observed: Optional[StandardScaler] = None,
        scaler_known: Optional[StandardScaler] = None,
        is_training: bool = False
    ):
        self.df = df.copy()
        self.config = feature_config
        self.stock_code = stock_code
        self.is_training = is_training

        # Static features (constant for all samples)
        self.sector_id = get_sector_id(stock_code)
        self.market_cap_group = get_market_cap_group(market_cap)

        # Add calendar features if not present
        self._add_calendar_features()

        # Filter available features
        self.available_observed = [f for f in feature_config.observed_features
                                   if f in self.df.columns]
        self.available_known = [f for f in feature_config.known_features
                               if f in self.df.columns]

        logger.info(f"Available observed features: {len(self.available_observed)}")
        logger.info(f"Available known features: {len(self.available_known)}")

        # Handle scalers
        self.scaler_observed = scaler_observed
        self.scaler_known = scaler_known

        if is_training:
            # Fit scalers on training data
            self._fit_scalers()
        elif scaler_observed is None or scaler_known is None:
            raise ValueError(
                "For non-training datasets, pre-fitted scalers must be provided "
                "to avoid data leakage."
            )

        # Prepare data arrays
        self._prepare_data()

        # Create sequences
        self._create_sequences()

    def _add_calendar_features(self):
        """Add calendar features to DataFrame if not present."""
        # Ensure index is datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'Date' in self.df.columns:
                self.df.set_index('Date', inplace=True)
            elif 'date' in self.df.columns:
                self.df.set_index('date', inplace=True)
            self.df.index = pd.to_datetime(self.df.index)

        # Add calendar features
        if 'day_of_week' not in self.df.columns:
            self.df['day_of_week'] = self.df.index.dayofweek / 6.0  # Normalize to 0-1

        if 'day_of_month' not in self.df.columns:
            self.df['day_of_month'] = self.df.index.day / 31.0  # Normalize to 0-1

        if 'month' not in self.df.columns:
            self.df['month'] = self.df.index.month / 12.0  # Normalize to 0-1

        if 'quarter' not in self.df.columns:
            self.df['quarter'] = self.df.index.quarter / 4.0  # Normalize to 0-1

        if 'is_month_end' not in self.df.columns:
            self.df['is_month_end'] = self.df.index.is_month_end.astype(float)

    def _fit_scalers(self):
        """Fit scalers on training data only."""
        # RobustScaler is better for financial data (handles outliers)
        if self.scaler_observed is None:
            self.scaler_observed = RobustScaler()
            observed_data = self.df[self.available_observed].values
            # Handle inf/nan
            observed_data = np.nan_to_num(observed_data, nan=0.0, posinf=0.0, neginf=0.0)
            self.scaler_observed.fit(observed_data)

        if self.scaler_known is None:
            self.scaler_known = StandardScaler()
            known_data = self.df[self.available_known].values
            known_data = np.nan_to_num(known_data, nan=0.0, posinf=0.0, neginf=0.0)
            self.scaler_known.fit(known_data)

    def _prepare_data(self):
        """Prepare and scale data arrays."""
        # Observed features
        observed_data = self.df[self.available_observed].values
        observed_data = np.nan_to_num(observed_data, nan=0.0, posinf=0.0, neginf=0.0)
        self.observed_scaled = self.scaler_observed.transform(observed_data)

        # Known features
        known_data = self.df[self.available_known].values
        known_data = np.nan_to_num(known_data, nan=0.0, posinf=0.0, neginf=0.0)
        self.known_scaled = self.scaler_known.transform(known_data)

        # Target: Returns (not raw prices) for better learning
        if self.config.target_returns:
            self.target = self.df[self.config.target_column].pct_change().fillna(0).values
            # Clip extreme returns
            self.target = np.clip(self.target, -0.2, 0.2)
        else:
            # Raw close prices (scaled)
            close_idx = self.available_observed.index('Close') if 'Close' in self.available_observed else 0
            self.target = self.observed_scaled[:, close_idx]

    def _create_sequences(self):
        """Create input/output sequences."""
        seq_len = self.config.sequence_length
        horizon = self.config.forecast_horizon
        n_samples = len(self.df) - seq_len - horizon + 1

        if n_samples <= 0:
            raise ValueError(
                f"Not enough data to create sequences. "
                f"Data length: {len(self.df)}, "
                f"Sequence length: {seq_len}, "
                f"Horizon: {horizon}"
            )

        # Pre-allocate arrays
        self.x_observed = np.zeros((n_samples, seq_len, len(self.available_observed)))
        self.x_known = np.zeros((n_samples, seq_len, len(self.available_known)))
        self.y = np.zeros((n_samples, horizon))
        self.y_returns = np.zeros((n_samples, horizon))  # Actual returns for Sharpe

        for i in range(n_samples):
            # Input sequence: t-seq_len to t
            self.x_observed[i] = self.observed_scaled[i:i + seq_len]
            self.x_known[i] = self.known_scaled[i:i + seq_len]

            # Target: next horizon returns
            self.y[i] = self.target[i + seq_len:i + seq_len + horizon]
            self.y_returns[i] = self.target[i + seq_len:i + seq_len + horizon]

        # Static features (same for all samples)
        self.x_static = np.array([
            [self.sector_id, self.market_cap_group]
        ] * n_samples, dtype=np.float32)

        logger.info(f"Created {n_samples} sequences")
        logger.info(f"  x_observed shape: {self.x_observed.shape}")
        logger.info(f"  x_known shape: {self.x_known.shape}")
        logger.info(f"  x_static shape: {self.x_static.shape}")
        logger.info(f"  y shape: {self.y.shape}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.

        Returns:
            Tuple of (x_observed, x_known, x_static, y, y_returns)
        """
        return (
            torch.tensor(self.x_observed[idx], dtype=torch.float32),
            torch.tensor(self.x_known[idx], dtype=torch.float32),
            torch.tensor(self.x_static[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.y_returns[idx], dtype=torch.float32),
        )

    def get_scalers(self) -> Tuple[StandardScaler, StandardScaler]:
        """Return fitted scalers for use with validation/test sets."""
        return self.scaler_observed, self.scaler_known

    def verify_shapes(self, config) -> bool:
        """
        Verify that data shapes match model expectations.

        Args:
            config: TFTConfig from model

        Returns:
            True if shapes are valid
        """
        # Check observed features dimension
        assert self.x_observed.shape[2] <= config.num_dynamic_features, \
            f"Observed features ({self.x_observed.shape[2]}) exceed model capacity ({config.num_dynamic_features})"

        # Check known features dimension
        assert self.x_known.shape[2] <= config.num_known_features, \
            f"Known features ({self.x_known.shape[2]}) exceed model capacity ({config.num_known_features})"

        # Check sequence length
        assert self.x_observed.shape[1] == config.sequence_length, \
            f"Sequence length mismatch: data ({self.x_observed.shape[1]}) != model ({config.sequence_length})"

        # Check forecast horizon
        assert self.y.shape[1] == config.forecast_horizon, \
            f"Forecast horizon mismatch: data ({self.y.shape[1]}) != model ({config.forecast_horizon})"

        logger.info("Shape verification passed!")
        return True


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def tft_collate_fn(batch):
    """
    Custom collate function for TFT DataLoader.

    Handles the 5-tuple output from TFTDataset.
    """
    x_observed = torch.stack([item[0] for item in batch])
    x_known = torch.stack([item[1] for item in batch])
    x_static = torch.stack([item[2] for item in batch])
    y = torch.stack([item[3] for item in batch])
    y_returns = torch.stack([item[4] for item in batch])

    return x_observed, x_known, x_static, y, y_returns


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def create_tft_dataloaders(
    df: pd.DataFrame,
    stock_code: str = '2222',
    market_cap: float = 1e12,
    feature_config: Optional[TFTFeatureConfig] = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test DataLoaders from a DataFrame.

    CRITICAL: Scalers are fit ONLY on training data to prevent data leakage.

    Args:
        df: DataFrame with all features and price data
        stock_code: Stock ticker for sector classification
        market_cap: Market capitalization in SAR
        feature_config: Optional custom feature configuration
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata_dict)
    """
    if feature_config is None:
        feature_config = TFTFeatureConfig()

    # Calculate minimum required data length
    # Need enough data so that even the smallest split (test) has at least 1 sequence
    min_seq_samples = feature_config.sequence_length + feature_config.forecast_horizon + 1
    test_split = 1 - train_split - val_split
    min_test_size = max(min_seq_samples, 10)  # At least 10 samples or enough for 1 sequence
    min_total_data = int(min_test_size / test_split) if test_split > 0 else min_seq_samples * 3

    if len(df) < min_seq_samples + 20:  # Absolute minimum: 1 sequence + buffer
        raise ValueError(
            f"Insufficient data for TFT training. "
            f"Have {len(df)} rows, need at least {min_seq_samples + 20} "
            f"(seq_len={feature_config.sequence_length}, horizon={feature_config.forecast_horizon})"
        )

    # Calculate minimum samples needed per split
    min_split_size = feature_config.sequence_length + feature_config.forecast_horizon + 1

    # Calculate split indices (chronological split)
    n_samples = len(df)

    # Calculate ideal splits
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))

    # Ensure minimum sizes for val and test
    # Work backwards from end to guarantee test has enough
    test_size = n_samples - val_end
    val_size = val_end - train_end

    if test_size < min_split_size:
        # Not enough test data - adjust val_end
        val_end = n_samples - min_split_size
        logger.warning(f"Adjusted test split: need {min_split_size}, now have {n_samples - val_end}")

    if (val_end - train_end) < min_split_size:
        # Not enough val data - reduce train
        train_end = val_end - min_split_size
        if train_end < min_split_size:
            # Not enough data overall - use minimal val/test
            train_end = n_samples - 2 * min_split_size
            val_end = n_samples - min_split_size
            logger.warning(f"Minimal splits: train ends at {train_end}")

    # Final validation
    if train_end < min_split_size:
        raise ValueError(
            f"Not enough data for proper train/val/test split. "
            f"Have {n_samples} rows, need at least {3 * min_split_size}"
        )

    logger.info(f"Data split: Train[0:{train_end}] ({train_end}), Val[{train_end}:{val_end}] ({val_end - train_end}), Test[{val_end}:{n_samples}] ({n_samples - val_end})")

    # Split data
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    # Create training dataset (fits scalers)
    train_dataset = TFTDataset(
        df=df_train,
        feature_config=feature_config,
        stock_code=stock_code,
        market_cap=market_cap,
        is_training=True
    )

    # Get fitted scalers from training set
    scaler_observed, scaler_known = train_dataset.get_scalers()

    # Create validation dataset (uses training scalers)
    val_dataset = TFTDataset(
        df=df_val,
        feature_config=feature_config,
        stock_code=stock_code,
        market_cap=market_cap,
        scaler_observed=scaler_observed,
        scaler_known=scaler_known,
        is_training=False
    )

    # Create test dataset (uses training scalers)
    test_dataset = TFTDataset(
        df=df_test,
        feature_config=feature_config,
        stock_code=stock_code,
        market_cap=market_cap,
        scaler_observed=scaler_observed,
        scaler_known=scaler_known,
        is_training=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=tft_collate_fn,
        drop_last=True  # Ensure consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=tft_collate_fn,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=tft_collate_fn,
        drop_last=False
    )

    # Metadata for model configuration
    metadata = {
        'num_observed_features': len(train_dataset.available_observed),
        'num_known_features': len(train_dataset.available_known),
        'num_static_features': 2,  # sector_id, market_cap_group
        'sequence_length': feature_config.sequence_length,
        'forecast_horizon': feature_config.forecast_horizon,
        'observed_feature_names': train_dataset.available_observed,
        'known_feature_names': train_dataset.available_known,
        'scaler_observed': scaler_observed,
        'scaler_known': scaler_known,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
    }

    logger.info(f"Created DataLoaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader, metadata


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def pad_or_trim_features(
    x_observed: torch.Tensor,
    x_known: torch.Tensor,
    target_observed: int,
    target_known: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adjust feature tensors to match model input dimensions.

    - If features < target: pad with zeros
    - If features > target: truncate to target size

    Args:
        x_observed: Observed features tensor (batch, seq, features)
        x_known: Known features tensor (batch, seq, features)
        target_observed: Target number of observed features
        target_known: Target number of known features

    Returns:
        Adjusted tensors matching target dimensions
    """
    batch, seq, obs_feat = x_observed.shape
    _, _, known_feat = x_known.shape

    # Adjust observed features
    if obs_feat < target_observed:
        # Pad with zeros
        padding = torch.zeros(batch, seq, target_observed - obs_feat, device=x_observed.device)
        x_observed = torch.cat([x_observed, padding], dim=-1)
    elif obs_feat > target_observed:
        # Truncate (keep first target_observed features)
        x_observed = x_observed[:, :, :target_observed]

    # Adjust known features
    if known_feat < target_known:
        # Pad with zeros
        padding = torch.zeros(batch, seq, target_known - known_feat, device=x_known.device)
        x_known = torch.cat([x_known, padding], dim=-1)
    elif known_feat > target_known:
        # Truncate
        x_known = x_known[:, :, :target_known]

    return x_observed, x_known


# Alias for backward compatibility
pad_features_to_model_size = pad_or_trim_features


def verify_batch_shapes(batch, model_config) -> bool:
    """
    Verify batch shapes match model expectations.

    Args:
        batch: Tuple from DataLoader
        model_config: TFTConfig from model

    Returns:
        True if all shapes are valid

    Raises:
        AssertionError: If shapes don't match
    """
    x_observed, x_known, x_static, y, y_returns = batch

    batch_size = x_observed.shape[0]

    # Static features: (batch, 2)
    assert x_static.shape == (batch_size, 2), \
        f"x_static shape {x_static.shape} != expected ({batch_size}, 2)"

    # Observed features: (batch, seq_len, num_dynamic_features)
    assert x_observed.shape[1] == model_config.sequence_length, \
        f"Sequence length {x_observed.shape[1]} != expected {model_config.sequence_length}"

    # Target: (batch, forecast_horizon)
    assert y.shape == (batch_size, model_config.forecast_horizon), \
        f"Target shape {y.shape} != expected ({batch_size}, {model_config.forecast_horizon})"

    logger.debug(f"Batch shapes verified: observed={x_observed.shape}, known={x_known.shape}, static={x_static.shape}, y={y.shape}")
    return True


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("TFT Data Loader - Test Suite")
    print("=" * 70)

    # Create synthetic test data
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')

    # Generate synthetic OHLCV
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)

    df = pd.DataFrame({
        'Date': dates,
        'Open': close + np.random.randn(n_days) * 0.2,
        'High': close + np.abs(np.random.randn(n_days) * 0.5),
        'Low': close - np.abs(np.random.randn(n_days) * 0.5),
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        # Add some indicator features
        'RSI': 50 + np.random.randn(n_days) * 15,
        'MACD': np.random.randn(n_days) * 0.5,
        'MACD_Signal': np.random.randn(n_days) * 0.3,
        'MACD_Histogram': np.random.randn(n_days) * 0.2,
        'ATR': np.abs(np.random.randn(n_days) * 1.5),
        'BB_Width': np.abs(np.random.randn(n_days) * 0.1),
        'Volatility': np.abs(np.random.randn(n_days) * 0.02),
        'HV_20': np.abs(np.random.randn(n_days) * 0.3),
        'SMA_20': close,
        'SMA_50': close,
        'EMA_12': close,
        'EMA_26': close,
        'OBV': np.cumsum(np.random.randn(n_days) * 1000000),
        'Volume_MA': 5000000 + np.random.randn(n_days) * 1000000,
        'Volume_Ratio': 1 + np.random.randn(n_days) * 0.3,
        'Relative_Volume': 1 + np.random.randn(n_days) * 0.3,
        'Stoch_K': 50 + np.random.randn(n_days) * 25,
        'Williams_R': -50 + np.random.randn(n_days) * 25,
        'ADX': 25 + np.abs(np.random.randn(n_days) * 15),
        'CCI': np.random.randn(n_days) * 100,
        'MFI': 50 + np.random.randn(n_days) * 20,
        'Ultimate_Osc': 50 + np.random.randn(n_days) * 15,
    })

    df.set_index('Date', inplace=True)

    print(f"\nSynthetic data shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}... ({len(df.columns)} total)")

    # Test feature config
    config = TFTFeatureConfig(
        sequence_length=60,
        forecast_horizon=5
    )

    print(f"\nFeature Config:")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Forecast horizon: {config.forecast_horizon}")
    print(f"  Static features: {config.static_features}")
    print(f"  Known features: {config.known_features}")
    print(f"  Observed features: {len(config.observed_features)} features")

    # Create DataLoaders
    print("\n" + "-" * 50)
    print("Creating DataLoaders...")

    train_loader, val_loader, test_loader, metadata = create_tft_dataloaders(
        df=df,
        stock_code='2222',
        market_cap=2e12,
        feature_config=config,
        train_split=0.7,
        val_split=0.15,
        batch_size=32
    )

    print(f"\nMetadata:")
    print(f"  Observed features: {metadata['num_observed_features']}")
    print(f"  Known features: {metadata['num_known_features']}")
    print(f"  Train samples: {metadata['train_samples']}")
    print(f"  Val samples: {metadata['val_samples']}")
    print(f"  Test samples: {metadata['test_samples']}")

    # Test batch
    print("\n" + "-" * 50)
    print("Testing batch retrieval...")

    batch = next(iter(train_loader))
    x_observed, x_known, x_static, y, y_returns = batch

    print(f"\nBatch shapes:")
    print(f"  x_observed: {x_observed.shape} (batch, seq_len, observed_features)")
    print(f"  x_known: {x_known.shape} (batch, seq_len, known_features)")
    print(f"  x_static: {x_static.shape} (batch, static_features)")
    print(f"  y: {y.shape} (batch, forecast_horizon)")
    print(f"  y_returns: {y_returns.shape} (batch, forecast_horizon)")

    print(f"\nStatic features sample (first 3):")
    print(f"  sector_id, market_cap_group: {x_static[:3].numpy()}")

    print(f"\nTarget returns sample (first 3):")
    print(f"  {y[:3].numpy()}")

    # Shape assertions
    print("\n" + "-" * 50)
    print("Running shape assertions...")

    assert x_observed.shape[0] == 32, "Batch size mismatch"
    assert x_observed.shape[1] == 60, "Sequence length mismatch"
    assert x_static.shape[1] == 2, "Static features mismatch"
    assert y.shape[1] == 5, "Forecast horizon mismatch"

    print("All shape assertions PASSED!")

    print("\n" + "=" * 70)
    print("TFT Data Loader Test: SUCCESS")
    print("=" * 70)
