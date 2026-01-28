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
# Official TASI Sector Classifications (Tadawul)
# ============================================================================

SAUDI_STOCKS = {
    # ========== ENERGY (قطاع الطاقة) ==========
    "2222": {"name": "Saudi Aramco", "sector": "Energy"},
    "2380": {"name": "Petro Rabigh", "sector": "Energy"},
    "4030": {"name": "Bahri", "sector": "Energy"},
    "2381": {"name": "SAFCO", "sector": "Energy"},
    "2382": {"name": "ADNOC Drilling", "sector": "Energy"},

    # ========== MATERIALS (قطاع المواد الأساسية) ==========
    # --- Chemicals ---
    "2010": {"name": "SABIC", "sector": "Materials"},
    "2350": {"name": "Saudi Kayan", "sector": "Materials"},
    "2310": {"name": "Sipchem", "sector": "Materials"},
    "2060": {"name": "Tasnee", "sector": "Materials"},
    "2210": {"name": "Nama Chemicals", "sector": "Materials"},
    "2250": {"name": "Chemanol", "sector": "Materials"},
    "2290": {"name": "Yansab", "sector": "Materials"},
    "2170": {"name": "Alujain", "sector": "Materials"},
    "2020": {"name": "SAPCO", "sector": "Materials"},
    "2330": {"name": "Advanced Petrochemical", "sector": "Materials"},
    "2001": {"name": "Chemanol", "sector": "Materials"},
    "2002": {"name": "Petrochem", "sector": "Materials"},

    # --- Metals & Mining ---
    "1211": {"name": "Ma'aden", "sector": "Materials"},
    "1301": {"name": "Astra Industrial", "sector": "Materials"},
    "1304": {"name": "Al Yamamah Steel", "sector": "Materials"},
    "1320": {"name": "SSAB", "sector": "Materials"},
    "1302": {"name": "Bawan", "sector": "Materials"},
    "1303": {"name": "Electrical Industries", "sector": "Materials"},
    "1321": {"name": "Saudi Steel Pipe", "sector": "Materials"},

    # --- Building Materials & Cement ---
    "1202": {"name": "MEPCO", "sector": "Materials"},
    "2090": {"name": "Jouf Cement", "sector": "Materials"},
    "3001": {"name": "Hail Cement", "sector": "Materials"},
    "3002": {"name": "Najran Cement", "sector": "Materials"},
    "3003": {"name": "City Cement", "sector": "Materials"},
    "3004": {"name": "Northern Region Cement", "sector": "Materials"},
    "3005": {"name": "Umm Al-Qura Cement", "sector": "Materials"},
    "3010": {"name": "Arabian Cement", "sector": "Materials"},
    "3020": {"name": "Yamama Cement", "sector": "Materials"},
    "3030": {"name": "Saudi Cement", "sector": "Materials"},
    "3040": {"name": "Qassim Cement", "sector": "Materials"},
    "3050": {"name": "Southern Province Cement", "sector": "Materials"},
    "3060": {"name": "Yanbu Cement", "sector": "Materials"},
    "3080": {"name": "Eastern Cement", "sector": "Materials"},
    "3090": {"name": "Tabuk Cement", "sector": "Materials"},
    "3091": {"name": "Al Jouf Cement", "sector": "Materials"},
    "2240": {"name": "Zamil Industrial", "sector": "Materials"},
    "2370": {"name": "MESC", "sector": "Materials"},

    # ========== BANKS (قطاع البنوك) ==========
    "1120": {"name": "Al Rajhi Bank", "sector": "Banks"},
    "1180": {"name": "Al Inma Bank", "sector": "Banks"},
    "1010": {"name": "Riyad Bank", "sector": "Banks"},
    "1020": {"name": "Bank AlJazira", "sector": "Banks"},
    "1030": {"name": "Saudi Investment Bank", "sector": "Banks"},
    "1050": {"name": "Banque Saudi Fransi", "sector": "Banks"},
    "1080": {"name": "Arab National Bank", "sector": "Banks"},
    "1090": {"name": "Saudi British Bank (SABB)", "sector": "Banks"},
    "1140": {"name": "Bank Albilad", "sector": "Banks"},
    "1150": {"name": "Alinma Bank", "sector": "Banks"},
    "1182": {"name": "Saudi National Bank (SNB)", "sector": "Banks"},

    # ========== DIVERSIFIED FINANCIALS (الخدمات المالية) ==========
    "1111": {"name": "Tadawul Group", "sector": "Diversified Financials"},
    "4280": {"name": "Kingdom Holding", "sector": "Diversified Financials"},
    "4130": {"name": "Al Rajhi Capital", "sector": "Diversified Financials"},
    "4081": {"name": "Al Moammar (ACES)", "sector": "Diversified Financials"},
    "1183": {"name": "SNB Capital", "sector": "Diversified Financials"},

    # ========== INSURANCE (قطاع التأمين) ==========
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

    # ========== TELECOMMUNICATION SERVICES (قطاع الاتصالات) ==========
    "7010": {"name": "STC", "sector": "Telecommunication Services"},
    "7020": {"name": "Mobily", "sector": "Telecommunication Services"},
    "7030": {"name": "Zain KSA", "sector": "Telecommunication Services"},
    "7040": {"name": "Dawiyat", "sector": "Telecommunication Services"},

    # ========== UTILITIES (قطاع المرافق العامة) ==========
    "5110": {"name": "Saudi Electricity", "sector": "Utilities"},
    "2082": {"name": "ACWA Power", "sector": "Utilities"},
    "2083": {"name": "Marafiq", "sector": "Utilities"},
    "2080": {"name": "GASCO", "sector": "Utilities"},
    "2081": {"name": "Alkhorayef Water", "sector": "Utilities"},
    "2084": {"name": "Engie Saudi", "sector": "Utilities"},

    # ========== REAL ESTATE MANAGEMENT & DEVELOPMENT (إدارة وتطوير العقارات) ==========
    "4300": {"name": "Dar Al Arkan", "sector": "Real Estate"},
    "4310": {"name": "Emaar Economic City", "sector": "Real Estate"},
    "4320": {"name": "Al Andalus Property", "sector": "Real Estate"},
    "4322": {"name": "Retal Urban Development", "sector": "Real Estate"},
    "4323": {"name": "Sumou Real Estate", "sector": "Real Estate"},
    "4020": {"name": "Saudi Real Estate", "sector": "Real Estate"},
    "4150": {"name": "Makkah Construction", "sector": "Real Estate"},
    "4220": {"name": "Emaar", "sector": "Real Estate"},
    "4250": {"name": "Jabal Omar", "sector": "Real Estate"},
    "4100": {"name": "Knowledge Economic City", "sector": "Real Estate"},
    "4230": {"name": "Red Sea International", "sector": "Real Estate"},

    # ========== REITs (صناديق الاستثمار العقارية المتداولة) ==========
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

    # ========== CONSUMER SERVICES (الخدمات الاستهلاكية) ==========
    # --- Retailing ---
    "4001": {"name": "Al Othaim Markets", "sector": "Consumer Services"},
    "4003": {"name": "Extra", "sector": "Consumer Services"},
    "4050": {"name": "SACO", "sector": "Consumer Services"},
    "4051": {"name": "BinDawood Holding", "sector": "Consumer Services"},
    "4190": {"name": "Jarir Marketing", "sector": "Consumer Services"},
    "4240": {"name": "Cenomi Retail", "sector": "Consumer Services"},

    # --- Hotels & Tourism ---
    "1810": {"name": "Seera Holding", "sector": "Consumer Services"},
    "1820": {"name": "Alhokair (Cenomi Centers)", "sector": "Consumer Services"},
    "1831": {"name": "Dur Hospitality", "sector": "Consumer Services"},
    "1832": {"name": "SHAMS", "sector": "Consumer Services"},
    "1830": {"name": "Leejam Sports", "sector": "Consumer Services"},
    "4170": {"name": "Al Tayyar Travel", "sector": "Consumer Services"},

    # --- Media & Entertainment ---
    "4070": {"name": "Tihama", "sector": "Consumer Services"},
    "4071": {"name": "MBC Group", "sector": "Consumer Services"},
    "4080": {"name": "Aseer Trading", "sector": "Consumer Services"},

    # ========== FOOD & STAPLES RETAILING (تجزئة الأغذية) ==========
    "4200": {"name": "Aldrees", "sector": "Food & Staples Retailing"},
    "4061": {"name": "Anaam Holding", "sector": "Food & Staples Retailing"},

    # ========== FOOD & BEVERAGES (الأغذية والمشروبات) ==========
    "2050": {"name": "Savola Group", "sector": "Food & Beverages"},
    "2100": {"name": "Wafrah", "sector": "Food & Beverages"},
    "2270": {"name": "SADAFCO", "sector": "Food & Beverages"},
    "2280": {"name": "Almarai", "sector": "Food & Beverages"},
    "2281": {"name": "Tanmiah Food", "sector": "Food & Beverages"},
    "6001": {"name": "Halwani Bros", "sector": "Food & Beverages"},
    "6002": {"name": "Herfy Food", "sector": "Food & Beverages"},
    "6004": {"name": "Catering Holding", "sector": "Food & Beverages"},
    "6010": {"name": "NADEC", "sector": "Food & Beverages"},
    "6012": {"name": "Takween Advanced", "sector": "Food & Beverages"},
    "6014": {"name": "Al Jouf Agricultural", "sector": "Food & Beverages"},
    "6015": {"name": "Americana Restaurants", "sector": "Food & Beverages"},
    "6020": {"name": "Jazan Development", "sector": "Food & Beverages"},
    "6040": {"name": "Tabuk Agricultural", "sector": "Food & Beverages"},
    "6050": {"name": "Saudi Fisheries", "sector": "Food & Beverages"},
    "6060": {"name": "Sharqia Development", "sector": "Food & Beverages"},
    "6070": {"name": "Bishah Development", "sector": "Food & Beverages"},
    "6090": {"name": "Jazan Energy", "sector": "Food & Beverages"},

    # ========== HEALTH CARE EQUIPMENT & SERVICES (الرعاية الصحية) ==========
    "4002": {"name": "Mouwasat Medical", "sector": "Health Care"},
    "4004": {"name": "Dallah Healthcare", "sector": "Health Care"},
    "4005": {"name": "National Medical Care", "sector": "Health Care"},
    "4006": {"name": "Al Hammadi", "sector": "Health Care"},
    "4007": {"name": "Al Nahdi Medical", "sector": "Health Care"},
    "4013": {"name": "Dr. Sulaiman Al Habib", "sector": "Health Care"},
    "4009": {"name": "SASCO", "sector": "Health Care"},

    # ========== PHARMA, BIOTECH & LIFE SCIENCES (الأدوية) ==========
    "4015": {"name": "Saudi Pharmaceutical", "sector": "Pharma"},
    "4016": {"name": "Jamjoom Pharma", "sector": "Pharma"},
    "4017": {"name": "Dawa Pharma", "sector": "Pharma"},
    "4014": {"name": "Middle East Healthcare", "sector": "Pharma"},
    "4018": {"name": "Alandalus Healthcare", "sector": "Pharma"},

    # ========== CAPITAL GOODS (السلع الرأسمالية) ==========
    "1201": {"name": "Takween", "sector": "Capital Goods"},
    "1210": {"name": "BCI", "sector": "Capital Goods"},
    "1212": {"name": "Astra Industrial", "sector": "Capital Goods"},
    "1213": {"name": "Al Hassan Shaker", "sector": "Capital Goods"},
    "1214": {"name": "Al Shaker", "sector": "Capital Goods"},
    "2040": {"name": "Saudi Ceramic", "sector": "Capital Goods"},
    "2110": {"name": "Saudi Cable", "sector": "Capital Goods"},
    "2120": {"name": "Saudi Advanced Industries", "sector": "Capital Goods"},
    "2130": {"name": "SIIG", "sector": "Capital Goods"},
    "2140": {"name": "Alahsa Development", "sector": "Capital Goods"},
    "2150": {"name": "Mubrad", "sector": "Capital Goods"},
    "2160": {"name": "Saudi Vitrified Clay", "sector": "Capital Goods"},
    "2180": {"name": "FPC", "sector": "Capital Goods"},
    "2190": {"name": "SISCO", "sector": "Capital Goods"},
    "2200": {"name": "Arabian Pipes", "sector": "Capital Goods"},
    "2220": {"name": "Maadaniyah", "sector": "Capital Goods"},
    "2230": {"name": "Saudi Chemical", "sector": "Capital Goods"},
    "2300": {"name": "Saudi Paper Manufacturing", "sector": "Capital Goods"},
    "2320": {"name": "Al Babtain Power", "sector": "Capital Goods"},
    "2340": {"name": "ALABDULLATIF", "sector": "Capital Goods"},
    "2360": {"name": "SVCP", "sector": "Capital Goods"},

    # ========== TRANSPORTATION (النقل) ==========
    "4030": {"name": "Bahri", "sector": "Transportation"},
    "4031": {"name": "SAPTCO", "sector": "Transportation"},
    "4040": {"name": "Saudi Ground Services", "sector": "Transportation"},
    "4210": {"name": "SPPC", "sector": "Transportation"},
    "4261": {"name": "Theeb Rent a Car", "sector": "Transportation"},
    "4262": {"name": "Lumi Rental", "sector": "Transportation"},
    "4263": {"name": "SAL Saudi Logistics", "sector": "Transportation"},
    "4260": {"name": "Budget Saudi", "sector": "Transportation"},
    "4270": {"name": "BUDGET UITC", "sector": "Transportation"},

    # ========== SOFTWARE & SERVICES (البرمجيات والخدمات) ==========
    "7200": {"name": "Solutions by STC", "sector": "Software & Services"},
    "7201": {"name": "Elm", "sector": "Software & Services"},
    "7202": {"name": "Thiqah", "sector": "Software & Services"},
    "7203": {"name": "Bayanat", "sector": "Software & Services"},
    "7204": {"name": "Tawasul", "sector": "Software & Services"},

    # ========== CONSUMER DURABLES & APPAREL (السلع المعمرة والملابس) ==========
    "4012": {"name": "Tihama Advertising", "sector": "Consumer Durables"},
    "4180": {"name": "Fitaihi Holding", "sector": "Consumer Durables"},
    "4160": {"name": "Thimar", "sector": "Consumer Durables"},
    "4008": {"name": "Fitness Time (Leejam)", "sector": "Consumer Durables"},
    "4011": {"name": "Saudi Marketing", "sector": "Consumer Durables"},
}

# Sector groupings for easy filtering (aligned with official TASI sectors)
SECTORS = {
    "Energy": ["2222", "2380", "2381", "2382"],
    "Materials": [
        "2010", "2350", "2310", "2060", "2210", "2250", "2290", "2170", "2001", "2020", "2330", "2002",
        "1211", "1301", "1304", "1320", "1302", "1303", "1321",
        "1202", "2090", "3001", "3002", "3003", "3004", "3005", "3010", "3020", "3030", "3040", "3050", "3060", "3080", "3090", "3091", "2240", "2370"
    ],
    "Banks": ["1120", "1180", "1010", "1020", "1030", "1050", "1080", "1090", "1140", "1150", "1182"],
    "Diversified Financials": ["1111", "4280", "4130", "4081", "1183"],
    "Insurance": [
        "8010", "8012", "8020", "8030", "8040", "8050", "8060", "8070", "8100", "8120",
        "8150", "8160", "8170", "8180", "8190", "8200", "8210", "8230", "8240", "8250",
        "8260", "8270", "8280", "8300", "8310", "8311"
    ],
    "Telecommunication Services": ["7010", "7020", "7030", "7040"],
    "Utilities": ["5110", "2082", "2083", "2080", "2081", "2084"],
    "Real Estate": ["4300", "4310", "4320", "4322", "4323", "4020", "4150", "4220", "4250", "4100", "4230"],
    "REITs": [
        "4330", "4331", "4332", "4333", "4334", "4335", "4336", "4337", "4338", "4339",
        "4340", "4342", "4344", "4345", "4346", "4347", "4348"
    ],
    "Consumer Services": [
        "4001", "4003", "4050", "4051", "4190", "4240",
        "1810", "1820", "1831", "1832", "1830", "4170",
        "4070", "4071", "4080"
    ],
    "Food & Staples Retailing": ["4200", "4061"],
    "Food & Beverages": [
        "2050", "2100", "2270", "2280", "2281", "6001", "6002", "6004", "6010", "6012",
        "6014", "6015", "6020", "6040", "6050", "6060", "6070", "6090"
    ],
    "Health Care": ["4002", "4004", "4005", "4006", "4007", "4013", "4009"],
    "Pharma": ["4015", "4016", "4017", "4014", "4018"],
    "Capital Goods": [
        "1201", "1210", "1212", "1213", "1214", "2040", "2110", "2120", "2130", "2140",
        "2150", "2160", "2180", "2190", "2200", "2220", "2230", "2300", "2320", "2340", "2360"
    ],
    "Transportation": ["4030", "4031", "4040", "4210", "4261", "4262", "4263", "4260", "4270"],
    "Software & Services": ["7200", "7201", "7202", "7203", "7204"],
    "Consumer Durables": ["4012", "4180", "4160", "4008", "4011"],
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
# ADVANCED LSTM CONFIGURATION - Simplified Architecture (Reduced Overfitting)
# ============================================================================
# Target Metrics: >75% Directional Accuracy, >70% Win Rate, >1.5 Sharpe Ratio
# Architecture simplified from 3 layers to 2 layers with higher dropout
ADVANCED_LSTM_CONFIG = {
    # Architecture (Simplified to reduce overfitting)
    "hidden_sizes": [64, 32],            # BiLSTM hidden sizes: 2 layers (was [128, 256, 128])
    "num_attention_heads": 4,            # Multi-head attention heads (kept same)
    "attention_dim": 64,                 # Dimension per attention head
    "bidirectional": True,               # Use bidirectional LSTM
    "use_residual": True,                # Residual connections between layers
    "use_layer_norm": True,              # Layer normalization
    "use_attention": True,               # Multi-head attention mechanism (kept same)

    # Regularization (Increased for robust learning)
    "dropout": 0.5,                      # Dropout rate (increased from 0.2)
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

    # Uncertainty (Monte Carlo Dropout)
    "output_uncertainty": True,          # Output uncertainty estimates
    "mc_dropout_samples": 10,            # MC Dropout samples (N=10 for speed/accuracy tradeoff)

    # Validation
    "use_purged_cv": True,               # Purged walk-forward CV
    "cv_purge_days": 5,                  # Days to purge between train/test
    "cv_embargo_days": 5,                # Days to embargo after test
    "cv_n_splits": 5,                    # Number of CV splits
}

# Advanced Features for High-Accuracy Model (40+ features)
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

    # Macroeconomic Features (NEW)
    "Oil_Correlation",      # 30-day rolling correlation with Brent Oil
    "Market_Trend",         # 14-day rolling return of TASI index
    "Oil_Momentum",         # 5-day change in oil price
    "Market_Volatility",    # 14-day rolling std of TASI returns (annualized)
    "Stock_Market_Beta",    # 30-day rolling beta vs TASI
]

# Macroeconomic Data Configuration
MACRO_CONFIG = {
    "brent_oil_symbol": "BZ=F",        # Brent Crude Oil Futures
    "tasi_index_symbol": "^TASI.SR",   # TASI Index
    "oil_correlation_window": 30,       # Days for oil correlation calculation
    "market_trend_window": 14,          # Days for market trend calculation
    "oil_momentum_window": 5,           # Days for oil momentum calculation
    "market_volatility_window": 14,     # Days for market volatility calculation
    "stock_market_beta_window": 30,     # Days for beta calculation
}

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
