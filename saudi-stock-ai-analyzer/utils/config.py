"""
Configuration settings and constants for Saudi Stock AI Analyzer
Version 3.0 - Advanced LSTM Edition (BiLSTM + Multi-Head Attention)
"""

import os

# Base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Saudi Stock Market Configuration
MARKET_SUFFIX = ".SR"

# Available Saudi Stocks for Demo
SAUDI_STOCKS = {
    "2222": {"name": "Saudi Aramco", "sector": "Energy"},
    "1120": {"name": "Al Rajhi Bank", "sector": "Banking"},
    "2010": {"name": "SABIC", "sector": "Chemicals"},
    "7010": {"name": "STC", "sector": "Telecommunications"},
    "1211": {"name": "Ma'aden", "sector": "Mining"},
    "2350": {"name": "Saudi Kayan", "sector": "Chemicals"},
    "1180": {"name": "Al Inma Bank", "sector": "Banking"},
    "2310": {"name": "Sipchem", "sector": "Chemicals"},
    "4030": {"name": "BAAN", "sector": "Real Estate"},
    "1010": {"name": "Riyad Bank", "sector": "Banking"},
    "2380": {"name": "PETRO RABIGH", "sector": "Energy"},
    "4200": {"name": "ALDREES", "sector": "Consumer"},
}

# Default stock for demo
DEFAULT_STOCK = "2222"

# Data Configuration
DEFAULT_PERIOD = "2y"  # Increased from 1y for more training data
DEFAULT_INTERVAL = "1d"

# Technical Indicator Parameters
INDICATORS = {
    "sma_short": 20,
    "sma_long": 50,
    "sma_very_long": 200,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_period": 14,
    "stochastic_period": 14,
    "stochastic_smooth": 3,
    "williams_period": 14,
    "adx_period": 14,
    "obv_smoothing": 20,
    "vwap_period": 14,
}

# RSI Thresholds (base values - adaptive thresholds adjust these)
# Optimized for higher win rate: more extreme values reduce false signals
RSI_OVERSOLD = 25
RSI_OVERBOUGHT = 75

# MACD Thresholds (for histogram)
MACD_THRESHOLD = 0.0

# ============================================================================
# ADVANCED LSTM CONFIGURATION - High Accuracy Model
# ============================================================================
# Target Metrics: >75% Directional Accuracy, >70% Win Rate, >1.5 Sharpe Ratio
ADVANCED_LSTM_CONFIG = {
    # Architecture
    "hidden_sizes": [128, 256, 128],    # BiLSTM hidden sizes per layer
    "num_attention_heads": 4,            # Multi-head attention heads
    "attention_dim": 64,                 # Dimension per attention head
    "bidirectional": True,               # Use bidirectional LSTM
    "use_residual": True,                # Residual connections between layers
    "use_layer_norm": True,              # Layer normalization
    "use_attention": True,               # Multi-head attention mechanism

    # Regularization
    "dropout": 0.2,                      # Dropout rate (lower than basic LSTM)
    "label_smoothing": 0.1,              # Label smoothing factor
    "weight_decay": 0.01,                # L2 regularization

    # Training
    "learning_rate": 0.0005,             # Initial learning rate
    "batch_size": 32,                    # Training batch size
    "epochs": 200,                       # Maximum training epochs
    "patience": 20,                      # Early stopping patience
    "lr_scheduler": "cosine_warm_restarts",  # LR scheduler type
    "lr_scheduler_T0": 10,               # Initial restart period
    "lr_scheduler_Tmult": 2,             # Period multiplier

    # Data
    "sequence_length": 60,               # Lookback window (60 trading days)
    "train_split": 0.7,                  # Training data ratio
    "val_split": 0.15,                   # Validation data ratio
    "test_split": 0.15,                  # Test data ratio

    # Augmentation
    "use_mixup": True,                   # Mixup data augmentation
    "mixup_alpha": 0.2,                  # Mixup alpha parameter
    "use_cutmix": False,                 # CutMix augmentation

    # Ensemble
    "n_ensemble_models": 5,              # Number of models in ensemble
    "ensemble_seeds": [42, 142, 242, 342, 442],  # Random seeds for diversity
    "ensemble_aggregation": "mean",      # Aggregation method

    # Uncertainty
    "output_uncertainty": True,          # Output uncertainty estimates
    "mc_dropout_samples": 100,           # MC Dropout samples for inference

    # Validation
    "use_purged_cv": True,               # Purged walk-forward CV
    "cv_purge_days": 5,                  # Days to purge between train/test
    "cv_embargo_days": 5,                # Days to embargo after test
    "cv_n_splits": 5,                    # Number of CV splits
}

# Advanced Features for High-Accuracy Model (35+ features)
ADVANCED_LSTM_FEATURES = [
    # Basic OHLCV
    "Close", "Volume", "High", "Low",

    # Moving Averages
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",

    # Momentum Indicators
    "RSI", "MACD", "MACD_Signal", "MACD_Histogram",

    # Volatility
    "ATR", "BB_Width", "Volatility", "HV_20",

    # Volume Indicators
    "OBV", "Volume_MA", "Volume_Ratio", "Relative_Volume",

    # Oscillators
    "Stoch_K", "Williams_R", "ADX", "CCI", "MFI", "Ultimate_Osc",

    # Microstructure
    "Volume_Momentum", "Price_Volume_Corr", "Amihud_Illiquidity", "Volume_Volatility",

    # Channels
    "Keltner_Position", "Donchian_Position",

    # Pattern Features
    "Gap_Percentage", "Body_Ratio", "Dist_From_20d_High", "Dist_From_20d_Low",

    # Statistical
    "Price_Zscore", "Price_Percentile", "Return_Skewness", "Vol_Regime",

    # Trend
    "ROC", "Momentum", "Daily_Return", "Choppiness", "TRIX",

    # Cross-Asset
    "Rolling_Beta", "Rolling_Alpha",
]

# Advanced Backtest Configuration
ADVANCED_BACKTEST_CONFIG = {
    "n_monte_carlo_simulations": 100,    # MC simulation count
    "slippage_range": [0.0005, 0.002],   # Min/max slippage
    "commission_range": [0.0005, 0.001], # Min/max commission
    "execution_delay_range": [0, 2],     # Min/max delay in bars
    "use_bootstrap": True,               # Block bootstrap for path variation
    "bootstrap_block_size": 10,          # Bootstrap block size
    "min_confidence_for_trade": 0.6,     # Minimum model confidence
}

# Trading Strategy Configuration
STRATEGY_CONFIG = {
    "min_confidence": 75,       # Minimum confidence for signal (0-100) - increased from 60 for higher win rate
    "position_size": 0.1,       # Default position size (10% of portfolio)
    "stop_loss": 0.05,          # Stop loss percentage (5%)
    "take_profit": 0.10,        # Take profit percentage (10%)
    "use_adaptive_thresholds": True,  # Adjust thresholds based on volatility
    "use_regime_detection": True,     # Detect market regime
    "hysteresis_periods": 3,          # Signal change delay periods
    "volume_confirmation": True,      # Require volume confirmation
    "require_multi_indicator": True,  # Require multiple indicator confirmation
    "min_confirming_indicators": 3,   # Minimum indicators that must agree
}

# Adaptive Threshold Configuration
ADAPTIVE_THRESHOLDS = {
    "rsi_base_oversold": 25,       # Updated from 30 for stricter signals
    "rsi_base_overbought": 75,     # Updated from 70 for stricter signals
    "rsi_adjustment_factor": 0.5,  # Volatility adjustment
    "macd_base_threshold": 0.0,
    "volatility_lookback": 20,     # Days to calculate volatility
    "min_rsi_threshold": 15,       # Minimum RSI threshold (more extreme)
    "max_rsi_threshold": 85,       # Maximum RSI threshold (more extreme)
}

# Regime Detection Configuration
REGIME_CONFIG = {
    "short_sma": 20,
    "long_sma": 50,
    "trend_strength_threshold": 0.02,  # 2% for trend classification
    "adx_trending_threshold": 25,      # ADX > 25 indicates trending
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    "initial_capital": 100000,  # Starting capital
    "position_size": 0.1,       # Position size as fraction of capital
    "commission": 0.001,        # Commission per trade (0.1%)
    "slippage": 0.001,          # Slippage estimate (0.1%)
    "default_hold_period": 10,  # Default holding period in days (increased from 5)
    "min_confidence": 75,       # Minimum confidence for trades (increased from 60)
    "use_stop_loss": True,      # Enable stop loss
    "use_take_profit": True,    # Enable take profit
}

# Risk Metrics Configuration
RISK_CONFIG = {
    "risk_free_rate": 0.02,     # Annual risk-free rate (2%)
    "target_return": 0.0,       # Target return for downside metrics
    "var_confidence_levels": [0.95, 0.99],  # VaR confidence levels
    "trading_days_per_year": 252,  # Trading days for annualization
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
    ],
}

# Model Paths
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models", "saved")
SIGNAL_HISTORY_PATH = os.path.join(BASE_PATH, "data", "signals")
SCALER_SAVE_PATH = os.path.join(BASE_PATH, "models", "scalers")

# Model Cache Configuration
CACHE_CONFIG = {
    "max_age": 86400,           # Maximum cache age in seconds (24 hours)
    "auto_cleanup": True,       # Automatically clean old cache files
    "cleanup_interval": 3600,   # Cleanup check interval (1 hour)
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(BASE_PATH, "logs", "analyzer.log"),
}

# Create required directories
for path in [MODEL_SAVE_PATH, SIGNAL_HISTORY_PATH, SCALER_SAVE_PATH]:
    os.makedirs(path, exist_ok=True)

# Create logs directory
logs_dir = os.path.join(BASE_PATH, "logs")
os.makedirs(logs_dir, exist_ok=True)
