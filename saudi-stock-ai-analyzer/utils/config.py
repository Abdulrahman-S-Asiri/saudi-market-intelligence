"""
Configuration settings and constants for Saudi Stock AI Analyzer
Version 3.0 - Advanced LSTM Edition (BiLSTM + Multi-Head Attention)
"""

import os

# Base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Saudi Stock Market Configuration
MARKET_SUFFIX = ".SR"

# ============================================================================
# TASI STOCKS BY SECTOR - Comprehensive Saudi Market Coverage
# ============================================================================

SAUDI_STOCKS = {
    # ========== ENERGY ==========
    "2222": {"name": "Saudi Aramco", "sector": "Energy"},
    "2380": {"name": "Petro Rabigh", "sector": "Energy"},
    "4030": {"name": "Bahri", "sector": "Energy"},
    "2030": {"name": "Sarco", "sector": "Energy"},
    "4031": {"name": "SAPTCO", "sector": "Energy"},

    # ========== MATERIALS - Chemicals ==========
    "2010": {"name": "SABIC", "sector": "Materials - Chemicals"},
    "2350": {"name": "Saudi Kayan", "sector": "Materials - Chemicals"},
    "2310": {"name": "Sipchem", "sector": "Materials - Chemicals"},
    "2060": {"name": "Tasnee", "sector": "Materials - Chemicals"},
    "2210": {"name": "Nama Chemicals", "sector": "Materials - Chemicals"},
    "2250": {"name": "Chemanol", "sector": "Materials - Chemicals"},
    "2290": {"name": "Yanbu National Petrochemical (Yansab)", "sector": "Materials - Chemicals"},
    "2170": {"name": "Alujain", "sector": "Materials - Chemicals"},
    "2001": {"name": "Methanol Chemicals (Chemanol)", "sector": "Materials - Chemicals"},
    "2020": {"name": "SAPCO", "sector": "Materials - Chemicals"},
    "2330": {"name": "Advanced Petrochemical", "sector": "Materials - Chemicals"},
    "2002": {"name": "Petrochem", "sector": "Materials - Chemicals"},

    # ========== MATERIALS - Mining & Metals ==========
    "1211": {"name": "Ma'aden", "sector": "Materials - Mining"},
    "1301": {"name": "Astra Industrial", "sector": "Materials - Mining"},
    "1304": {"name": "Al Yamamah Steel", "sector": "Materials - Mining"},
    "1320": {"name": "SSAB", "sector": "Materials - Mining"},
    "1302": {"name": "Bawan", "sector": "Materials - Mining"},
    "1303": {"name": "Electrical Industries", "sector": "Materials - Mining"},

    # ========== MATERIALS - Building & Construction ==========
    "1202": {"name": "Middle East Paper (MEPCO)", "sector": "Materials - Building"},
    "2090": {"name": "Jouf Cement", "sector": "Materials - Building"},
    "3001": {"name": "Hail Cement", "sector": "Materials - Building"},
    "3002": {"name": "Najran Cement", "sector": "Materials - Building"},
    "3003": {"name": "City Cement", "sector": "Materials - Building"},
    "3004": {"name": "Northern Region Cement", "sector": "Materials - Building"},
    "3005": {"name": "Umm Al-Qura Cement", "sector": "Materials - Building"},
    "3010": {"name": "Arabian Cement", "sector": "Materials - Building"},
    "3020": {"name": "Yamama Cement", "sector": "Materials - Building"},
    "3030": {"name": "Saudi Cement", "sector": "Materials - Building"},
    "3040": {"name": "Qassim Cement", "sector": "Materials - Building"},
    "3050": {"name": "Southern Province Cement", "sector": "Materials - Building"},
    "3060": {"name": "Yanbu Cement", "sector": "Materials - Building"},
    "3080": {"name": "Eastern Cement", "sector": "Materials - Building"},
    "3090": {"name": "Tabuk Cement", "sector": "Materials - Building"},
    "3091": {"name": "Al Jouf Cement", "sector": "Materials - Building"},
    "2240": {"name": "Zamil Industrial", "sector": "Materials - Building"},
    "2370": {"name": "MESC", "sector": "Materials - Building"},
    "1321": {"name": "Saudi Steel Pipe", "sector": "Materials - Building"},

    # ========== BANKS ==========
    "1120": {"name": "Al Rajhi Bank", "sector": "Banks"},
    "1180": {"name": "Al Inma Bank", "sector": "Banks"},
    "1010": {"name": "Riyad Bank", "sector": "Banks"},
    "1020": {"name": "Bank AlJazira", "sector": "Banks"},
    "1030": {"name": "Saudi Investment Bank (SAIB)", "sector": "Banks"},
    "1050": {"name": "BSF (Banque Saudi Fransi)", "sector": "Banks"},
    "1060": {"name": "SAMBA (now SNB)", "sector": "Banks"},
    "1080": {"name": "Arab National Bank (ANB)", "sector": "Banks"},
    "1090": {"name": "Saudi British Bank (SABB)", "sector": "Banks"},
    "1140": {"name": "Bank Albilad", "sector": "Banks"},
    "1150": {"name": "Alinma Bank", "sector": "Banks"},
    "1182": {"name": "SNB (Saudi National Bank)", "sector": "Banks"},

    # ========== DIVERSIFIED FINANCIALS ==========
    "1111": {"name": "Tadawul Group", "sector": "Diversified Financials"},
    "4280": {"name": "Kingdom Holding", "sector": "Diversified Financials"},
    "4081": {"name": "Naqi Water", "sector": "Diversified Financials"},
    "4082": {"name": "Al Moammar (ACES)", "sector": "Diversified Financials"},

    # ========== INSURANCE ==========
    "8010": {"name": "Tawuniya", "sector": "Insurance"},
    "8012": {"name": "Jazira Takaful", "sector": "Insurance"},
    "8020": {"name": "Malath Insurance", "sector": "Insurance"},
    "8030": {"name": "MedGulf", "sector": "Insurance"},
    "8040": {"name": "Alahlia Insurance", "sector": "Insurance"},
    "8050": {"name": "Salama Insurance", "sector": "Insurance"},
    "8060": {"name": "Walaa Insurance", "sector": "Insurance"},
    "8070": {"name": "Arabian Shield", "sector": "Insurance"},
    "8100": {"name": "SAICO", "sector": "Insurance"},
    "8120": {"name": "Gulf Union Insurance", "sector": "Insurance"},
    "8150": {"name": "ACIG", "sector": "Insurance"},
    "8160": {"name": "Alahli Takaful", "sector": "Insurance"},
    "8170": {"name": "Alinma Tokio Marine", "sector": "Insurance"},
    "8180": {"name": "Alrajhi Takaful", "sector": "Insurance"},
    "8190": {"name": "UCA", "sector": "Insurance"},
    "8200": {"name": "Al Sagr Insurance", "sector": "Insurance"},
    "8210": {"name": "Bupa Arabia", "sector": "Insurance"},
    "8230": {"name": "AXA Cooperative", "sector": "Insurance"},
    "8240": {"name": "AICC", "sector": "Insurance"},
    "8250": {"name": "GIG Saudi", "sector": "Insurance"},
    "8260": {"name": "Gulf General Insurance", "sector": "Insurance"},
    "8270": {"name": "Buruj Insurance", "sector": "Insurance"},
    "8280": {"name": "Al Alamiya Insurance", "sector": "Insurance"},
    "8300": {"name": "Wataniya Insurance", "sector": "Insurance"},
    "8310": {"name": "Amana Insurance", "sector": "Insurance"},
    "8311": {"name": "Rasan", "sector": "Insurance"},

    # ========== TELECOMMUNICATIONS ==========
    "7010": {"name": "STC (Saudi Telecom)", "sector": "Telecommunication Services"},
    "7020": {"name": "Etihad Etisalat (Mobily)", "sector": "Telecommunication Services"},
    "7030": {"name": "Zain KSA", "sector": "Telecommunication Services"},
    "7040": {"name": "Dawiyat", "sector": "Telecommunication Services"},

    # ========== UTILITIES ==========
    "5110": {"name": "Saudi Electricity (SEC)", "sector": "Utilities"},
    "2082": {"name": "ACWA Power", "sector": "Utilities"},
    "2083": {"name": "Marafiq", "sector": "Utilities"},
    "2080": {"name": "Gas & Industrialization (GASCO)", "sector": "Utilities"},
    "2081": {"name": "Al Toukhi", "sector": "Utilities"},
    "2084": {"name": "Engie Saudi", "sector": "Utilities"},

    # ========== REAL ESTATE ==========
    "4300": {"name": "Dar Al Arkan", "sector": "Real Estate Development"},
    "4310": {"name": "Emaar The Economic City", "sector": "Real Estate Development"},
    "4320": {"name": "Al Andalus Property", "sector": "Real Estate Development"},
    "4322": {"name": "Retal Urban Development", "sector": "Real Estate Development"},
    "4323": {"name": "Sumou Real Estate", "sector": "Real Estate Development"},
    "4020": {"name": "Al Akaria (Saudi Real Estate)", "sector": "Real Estate Development"},
    "4150": {"name": "Makkah Construction", "sector": "Real Estate Development"},
    "4220": {"name": "Emaar", "sector": "Real Estate Development"},
    "4250": {"name": "Jabal Omar Development", "sector": "Real Estate Development"},

    # ========== REITs ==========
    "4330": {"name": "Riyad REIT", "sector": "REITs"},
    "4331": {"name": "Jadwa REIT Al Haramain", "sector": "REITs"},
    "4332": {"name": "Jadwa REIT Saudi", "sector": "REITs"},
    "4333": {"name": "Taleem REIT", "sector": "REITs"},
    "4334": {"name": "Al Maather REIT", "sector": "REITs"},
    "4335": {"name": "Musharaka REIT", "sector": "REITs"},
    "4336": {"name": "Mulkia REIT", "sector": "REITs"},
    "4337": {"name": "SICO Saudi REIT", "sector": "REITs"},
    "4338": {"name": "Al Rajhi REIT", "sector": "REITs"},
    "4339": {"name": "Derayah REIT", "sector": "REITs"},
    "4340": {"name": "Al Jazira Mawten REIT", "sector": "REITs"},
    "4342": {"name": "Jadwa REIT CitiCenter", "sector": "REITs"},
    "4344": {"name": "Sedco Capital REIT", "sector": "REITs"},
    "4345": {"name": "Alinma Retail REIT", "sector": "REITs"},
    "4346": {"name": "MEFIC REIT", "sector": "REITs"},
    "4347": {"name": "Bonyan REIT", "sector": "REITs"},
    "4348": {"name": "Al Ahli REIT 1", "sector": "REITs"},

    # ========== RETAILING ==========
    "4001": {"name": "Abdullah Al Othaim Markets", "sector": "Retailing"},
    "4002": {"name": "Mouwasat Medical Services", "sector": "Retailing"},
    "4003": {"name": "Extra (United Electronics)", "sector": "Retailing"},
    "4004": {"name": "Dallah Healthcare", "sector": "Retailing"},
    "4005": {"name": "National Medical Care", "sector": "Retailing"},
    "4006": {"name": "Al Hammadi Holding", "sector": "Retailing"},
    "4007": {"name": "Al Nahdi Medical", "sector": "Retailing"},
    "4008": {"name": "Leejam Sports (Fitness Time)", "sector": "Retailing"},
    "4009": {"name": "Saudi Automotive Services (SASCO)", "sector": "Retailing"},
    "4050": {"name": "SACO", "sector": "Retailing"},
    "4051": {"name": "BinDawood Holding", "sector": "Retailing"},
    "4190": {"name": "Jarir Marketing", "sector": "Retailing"},
    "4200": {"name": "Aldrees Petroleum", "sector": "Retailing"},
    "4240": {"name": "Fawaz Alhokair (Cenomi)", "sector": "Retailing"},
    "4260": {"name": "Budget Saudi", "sector": "Retailing"},
    "4270": {"name": "United International Transportation (BUDGET)", "sector": "Retailing"},

    # ========== FOOD & BEVERAGES ==========
    "2050": {"name": "Savola Group", "sector": "Food & Beverages"},
    "2100": {"name": "Wafrah (Nadec)", "sector": "Food & Beverages"},
    "2270": {"name": "Saudia Dairy (SADAFCO)", "sector": "Food & Beverages"},
    "2280": {"name": "Almarai", "sector": "Food & Beverages"},
    "2281": {"name": "Tanmiah Food", "sector": "Food & Beverages"},
    "6001": {"name": "Halwani Bros", "sector": "Food & Beverages"},
    "6002": {"name": "Herfy Food Services", "sector": "Food & Beverages"},
    "6004": {"name": "Catering Holding (CATERING)", "sector": "Food & Beverages"},
    "6010": {"name": "National Agricultural Development (NADEC)", "sector": "Food & Beverages"},
    "6012": {"name": "Takween", "sector": "Food & Beverages"},
    "6013": {"name": "Theeb Rent a Car", "sector": "Food & Beverages"},
    "6014": {"name": "Al Jouf Agricultural", "sector": "Food & Beverages"},
    "6015": {"name": "Americana Restaurants", "sector": "Food & Beverages"},
    "6020": {"name": "Jazan Development", "sector": "Food & Beverages"},
    "6040": {"name": "Tabuk Agricultural Development", "sector": "Food & Beverages"},
    "6050": {"name": "Saudi Fisheries", "sector": "Food & Beverages"},
    "6060": {"name": "Sharqia Development", "sector": "Food & Beverages"},
    "6070": {"name": "Bishah Development", "sector": "Food & Beverages"},
    "6090": {"name": "Jazan Energy", "sector": "Food & Beverages"},

    # ========== CONSUMER DURABLES & APPAREL ==========
    "4011": {"name": "Al Sagr Insurance", "sector": "Consumer Durables"},
    "4012": {"name": "Tihama Advertising", "sector": "Consumer Durables"},
    "4180": {"name": "Fitaihi Holding", "sector": "Consumer Durables"},
    "4160": {"name": "Thimar", "sector": "Consumer Durables"},
    "4061": {"name": "Anaam Holding", "sector": "Consumer Durables"},

    # ========== HEALTHCARE ==========
    "4002": {"name": "Mouwasat Medical", "sector": "Healthcare"},
    "4004": {"name": "Dallah Healthcare", "sector": "Healthcare"},
    "4005": {"name": "National Medical Care", "sector": "Healthcare"},
    "4006": {"name": "Al Hammadi", "sector": "Healthcare"},
    "4007": {"name": "Al Nahdi Medical", "sector": "Healthcare"},
    "4013": {"name": "Sulaiman Al Habib Medical", "sector": "Healthcare"},
    "4014": {"name": "Al Mowasat", "sector": "Healthcare"},

    # ========== PHARMA ==========
    "4015": {"name": "Saudi Pharmaceutical", "sector": "Pharma"},
    "4016": {"name": "Jamjoom Pharma", "sector": "Pharma"},
    "4017": {"name": "Dawa Pharma", "sector": "Pharma"},

    # ========== CAPITAL GOODS & INDUSTRIALS ==========
    "1201": {"name": "Takween", "sector": "Capital Goods"},
    "1210": {"name": "BCI (Building Components)", "sector": "Capital Goods"},
    "1212": {"name": "Astra Industrial", "sector": "Capital Goods"},
    "1213": {"name": "Al Hassan Shaker", "sector": "Capital Goods"},
    "1214": {"name": "Al Shaker", "sector": "Capital Goods"},
    "2040": {"name": "Saudi Ceramic", "sector": "Capital Goods"},
    "2110": {"name": "Saudi Cable", "sector": "Capital Goods"},
    "2120": {"name": "Saudi Advanced Industries", "sector": "Capital Goods"},
    "2130": {"name": "Saudi Industrial Investment (SIIG)", "sector": "Capital Goods"},
    "2140": {"name": "Alahsa Development", "sector": "Capital Goods"},
    "2150": {"name": "Saudi Transport (Mubrad)", "sector": "Capital Goods"},
    "2160": {"name": "Saudi Vitrified Clay Pipe", "sector": "Capital Goods"},
    "2180": {"name": "Filing & Packing (FPC)", "sector": "Capital Goods"},
    "2190": {"name": "SISCO (Saudi Industrial Services)", "sector": "Capital Goods"},
    "2200": {"name": "Arabian Pipes", "sector": "Capital Goods"},
    "2220": {"name": "Maadaniyah", "sector": "Capital Goods"},
    "2230": {"name": "Saudi Chemical", "sector": "Capital Goods"},
    "2300": {"name": "Saudi Paper Manufacturing", "sector": "Capital Goods"},
    "2320": {"name": "Al Babtain Power", "sector": "Capital Goods"},
    "2340": {"name": "Al Abdullatif Industrial (ALABDULLATIF)", "sector": "Capital Goods"},
    "2360": {"name": "Saudi Vitrified Clay", "sector": "Capital Goods"},
    "4140": {"name": "Saudi Industrial Export", "sector": "Capital Goods"},
    "4141": {"name": "Al-Omran Industrial Trading", "sector": "Capital Goods"},
    "4142": {"name": "Alujain", "sector": "Capital Goods"},

    # ========== TRANSPORTATION ==========
    "4040": {"name": "Saudi Ground Services (SGS)", "sector": "Transportation"},
    "4110": {"name": "Mobily", "sector": "Transportation"},
    "4210": {"name": "Saudi Printing & Packaging (SPPC)", "sector": "Transportation"},
    "4261": {"name": "Theeb Rent a Car", "sector": "Transportation"},
    "4262": {"name": "Lumi Rental", "sector": "Transportation"},
    "4263": {"name": "SAL (Saudi Logistics)", "sector": "Transportation"},

    # ========== MEDIA & ENTERTAINMENT ==========
    "4070": {"name": "Tihama", "sector": "Media"},
    "4071": {"name": "MBC Group", "sector": "Media"},
    "4080": {"name": "Aseer Trading", "sector": "Media"},

    # ========== SOFTWARE & IT ==========
    "7200": {"name": "Solutions by STC", "sector": "Software & Services"},
    "7201": {"name": "Elm", "sector": "Software & Services"},
    "7202": {"name": "Thiqah", "sector": "Software & Services"},
    "7203": {"name": "Bayanat", "sector": "Software & Services"},
    "7204": {"name": "Tawasul", "sector": "Software & Services"},

    # ========== HOTELS & TOURISM ==========
    "1810": {"name": "Seera Holding", "sector": "Hotels & Tourism"},
    "1820": {"name": "Al Tayyar Travel", "sector": "Hotels & Tourism"},
    "1830": {"name": "Leejam Sports", "sector": "Hotels & Tourism"},
    "1831": {"name": "Dur Hospitality", "sector": "Hotels & Tourism"},
    "1832": {"name": "Tourism Enterprise (SHAMS)", "sector": "Hotels & Tourism"},
    "4170": {"name": "Al Tayyar", "sector": "Hotels & Tourism"},
}

# Sector groupings for easy filtering
SECTORS = {
    "Energy": ["2222", "2380", "4030", "2030", "4031"],
    "Materials - Chemicals": ["2010", "2350", "2310", "2060", "2210", "2250", "2290", "2170", "2001", "2020", "2330", "2002"],
    "Materials - Mining": ["1211", "1301", "1304", "1320", "1302", "1303"],
    "Materials - Building": ["1202", "2090", "3001", "3002", "3003", "3004", "3005", "3010", "3020", "3030", "3040", "3050", "3060", "3080", "3090", "3091", "2240", "2370", "1321"],
    "Banks": ["1120", "1180", "1010", "1020", "1030", "1050", "1060", "1080", "1090", "1140", "1150", "1182"],
    "Diversified Financials": ["1111", "4280", "4081", "4082"],
    "Insurance": ["8010", "8012", "8020", "8030", "8040", "8050", "8060", "8070", "8100", "8120", "8150", "8160", "8170", "8180", "8190", "8200", "8210", "8230", "8240", "8250", "8260", "8270", "8280", "8300", "8310", "8311"],
    "Telecommunication Services": ["7010", "7020", "7030", "7040"],
    "Utilities": ["5110", "2082", "2083", "2080", "2081", "2084"],
    "Real Estate Development": ["4300", "4310", "4320", "4322", "4323", "4020", "4150", "4220", "4250"],
    "REITs": ["4330", "4331", "4332", "4333", "4334", "4335", "4336", "4337", "4338", "4339", "4340", "4342", "4344", "4345", "4346", "4347", "4348"],
    "Retailing": ["4001", "4002", "4003", "4004", "4005", "4006", "4007", "4008", "4009", "4050", "4051", "4190", "4200", "4240", "4260", "4270"],
    "Food & Beverages": ["2050", "2100", "2270", "2280", "2281", "6001", "6002", "6004", "6010", "6012", "6013", "6014", "6015", "6020", "6040", "6050", "6060", "6070", "6090"],
    "Healthcare": ["4002", "4004", "4005", "4006", "4007", "4013", "4014"],
    "Pharma": ["4015", "4016", "4017"],
    "Capital Goods": ["1201", "1210", "1212", "1213", "1214", "2040", "2110", "2120", "2130", "2140", "2150", "2160", "2180", "2190", "2200", "2220", "2230", "2300", "2320", "2340", "2360", "4140", "4141", "4142"],
    "Transportation": ["4040", "4110", "4210", "4261", "4262", "4263"],
    "Media": ["4070", "4071", "4080"],
    "Software & Services": ["7200", "7201", "7202", "7203", "7204"],
    "Hotels & Tourism": ["1810", "1820", "1830", "1831", "1832", "4170"],
    "Consumer Durables": ["4011", "4012", "4180", "4160", "4061"],
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
