"""
Data preprocessing and cleaning module for Saudi Stock AI Analyzer
Includes technical indicators calculation and LSTM data preparation
FIXED: Data leakage issue - scaler now fits only on training data

Enhanced with 35+ features for high-accuracy LSTM model:
- Market Microstructure: Volume_Momentum, Price_Volume_Correlation, Amihud_Illiquidity
- Advanced Indicators: Keltner Channels, Donchian Channels, CCI, MFI, CMF, TRIX
- Pattern Features: Higher_High, Lower_Low, Inside_Bar, Gap_Percentage
- Cross-Asset: TASI_Index_Return, Relative_Strength_vs_TASI
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import INDICATORS, ADVANCED_LSTM_CONFIG


class DataPreprocessor:
    """
    Preprocesses stock data for analysis and LSTM model training
    FIXED: Scaler now only fits on training data to prevent data leakage
    """

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_fitted = False

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw stock data

        Args:
            df: Raw stock DataFrame from data_loader

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Check for column variations
        # Note: Skip macro columns (Oil_Close, TASI_Close) to preserve them
        col_mapping = {}
        for col in df.columns:
            col_str = str(col)
            # Skip macro columns - they should not be renamed
            if col_str in ['Oil_Close', 'TASI_Close']:
                continue
            if 'Date' in col_str or 'date' in col_str.lower():
                col_mapping[col] = 'Date'
            elif 'Open' in col_str:
                col_mapping[col] = 'Open'
            elif 'High' in col_str:
                col_mapping[col] = 'High'
            elif 'Low' in col_str:
                col_mapping[col] = 'Low'
            elif 'Close' in col_str and 'Adj' not in col_str:
                col_mapping[col] = 'Close'
            elif 'Volume' in col_str:
                col_mapping[col] = 'Volume'

        df = df.rename(columns=col_mapping)

        # Handle missing values
        df = df.ffill()
        df = df.bfill()

        # Remove rows with any remaining NaN
        df = df.dropna()

        # Ensure numeric types for standard columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Oil_Close', 'TASI_Close']
        for col in numeric_cols:
            if col in df.columns:
                # Check if column is a Series (not a DataFrame from duplicate columns)
                col_data = df[col]
                if isinstance(col_data, pd.DataFrame):
                    # If it's a DataFrame, take the first column
                    df[col] = col_data.iloc[:, 0]
                    col_data = df[col]

                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(col_data):
                    continue

                # Try to convert to numeric
                df[col] = pd.to_numeric(col_data, errors='coerce')

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the DataFrame

        Args:
            df: Cleaned stock DataFrame

        Returns:
            DataFrame with technical indicators added
        """
        df = df.copy()

        # ===== Moving Averages =====
        df['SMA_20'] = df['Close'].rolling(window=INDICATORS['sma_short']).mean()
        df['SMA_50'] = df['Close'].rolling(window=INDICATORS['sma_long']).mean()

        if len(df) >= INDICATORS['sma_very_long']:
            df['SMA_200'] = df['Close'].rolling(window=INDICATORS['sma_very_long']).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # ===== RSI (Relative Strength Index) =====
        df['RSI'] = self._calculate_rsi(df['Close'], INDICATORS['rsi_period'])

        # ===== MACD =====
        macd, signal, histogram = self._calculate_macd(
            df['Close'],
            INDICATORS['macd_fast'],
            INDICATORS['macd_slow'],
            INDICATORS['macd_signal']
        )
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram

        # ===== Bollinger Bands =====
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        # ===== ATR (Average True Range) =====
        df['ATR'] = self._calculate_atr(df, period=14)

        # ===== OBV (On-Balance Volume) =====
        df['OBV'] = self._calculate_obv(df)

        # ===== Stochastic Oscillator =====
        stoch_k, stoch_d = self._calculate_stochastic(df, k_period=14, d_period=3)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d

        # ===== Williams %R =====
        df['Williams_R'] = self._calculate_williams_r(df, period=14)

        # ===== ADX (Average Directional Index) =====
        df['ADX'] = self._calculate_adx(df, period=14)

        # ===== Price Rate of Change =====
        df['ROC'] = self._calculate_roc(df['Close'], period=10)

        # ===== Daily Returns =====
        df['Daily_Return'] = df['Close'].pct_change()

        # ===== Volatility (20-day rolling std of returns) =====
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

        # ===== Price momentum =====
        df['Momentum'] = df['Close'] - df['Close'].shift(10)

        # ===== Volume indicators =====
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # ===== VWAP (Volume Weighted Average Price) =====
        df['VWAP'] = self._calculate_vwap(df)

        # ===== Ichimoku Cloud Components =====
        ichimoku = self._calculate_ichimoku(df)
        df['Ichimoku_Tenkan'] = ichimoku['tenkan']
        df['Ichimoku_Kijun'] = ichimoku['kijun']
        df['Ichimoku_Senkou_A'] = ichimoku['senkou_a']
        df['Ichimoku_Senkou_B'] = ichimoku['senkou_b']

        # Drop NaN rows only for required indicators
        required_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
        available_required = [col for col in required_cols if col in df.columns]
        df = df.dropna(subset=available_required)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        exp_fast = prices.ewm(span=fast, adjust=False).mean()
        exp_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = exp_fast - exp_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)

    def _calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()

        stoch_k = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return stoch_k, stoch_d

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()

        williams_r = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
        return williams_r

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        tr = self._calculate_atr(df, period=1) * period

        plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_roc(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change"""
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap

    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line): 9-period high + low / 2
        tenkan = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2

        # Kijun-sen (Base Line): 26-period high + low / 2
        kijun = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2

        # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods
        senkou_a = ((tenkan + kijun) / 2).shift(26)

        # Senkou Span B (Leading Span B): 52-period high + low / 2, shifted 26 periods
        senkou_b = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)

        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b
        }

    # ========================================================================
    # ADVANCED FEATURES FOR HIGH-ACCURACY LSTM MODEL
    # ========================================================================

    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced features for high-accuracy LSTM model.

        This method adds 15+ new features including:
        - Market Microstructure indicators
        - Advanced technical indicators
        - Price pattern features
        - Cross-asset features

        Args:
            df: DataFrame with basic technical indicators already added

        Returns:
            DataFrame with advanced features added (~35 total features)
        """
        df = df.copy()

        # ===== MARKET MICROSTRUCTURE FEATURES =====
        df = self._add_microstructure_features(df)

        # ===== ADVANCED TECHNICAL INDICATORS =====
        df = self._add_advanced_indicators(df)

        # ===== PRICE PATTERN FEATURES =====
        df = self._add_pattern_features(df)

        # ===== CROSS-ASSET / RELATIVE FEATURES =====
        df = self._add_cross_asset_features(df)

        # ===== STATISTICAL FEATURES =====
        df = self._add_statistical_features(df)

        # Fill NaN with forward fill then backward fill
        df = df.ffill().bfill()

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features.

        These features capture market liquidity and trading activity patterns
        that are often predictive of short-term price movements.
        """
        # Volume Momentum (rate of change in volume)
        df['Volume_Momentum'] = df['Volume'].pct_change(5).fillna(0)

        # Price-Volume Correlation (rolling 20-day)
        df['Price_Volume_Corr'] = (
            df['Close'].rolling(window=20)
            .corr(df['Volume'])
            .fillna(0)
        )

        # Amihud Illiquidity Ratio (absolute return / dollar volume)
        # Higher values indicate less liquid stocks
        dollar_volume = df['Close'] * df['Volume']
        df['Amihud_Illiquidity'] = (
            (df['Close'].pct_change().abs() / (dollar_volume + 1))
            .rolling(window=20).mean()
            .fillna(0)
        )
        # Normalize to reasonable range
        df['Amihud_Illiquidity'] = df['Amihud_Illiquidity'] * 1e9

        # Volume Volatility (rolling std of volume)
        df['Volume_Volatility'] = (
            df['Volume'].rolling(window=20).std() /
            df['Volume'].rolling(window=20).mean()
        ).fillna(0)

        # Relative Volume (current vs average)
        df['Relative_Volume'] = (
            df['Volume'] / df['Volume'].rolling(window=20).mean()
        ).fillna(1)

        # Price Impact (price change per unit volume)
        df['Price_Impact'] = (
            df['Close'].pct_change().abs() /
            (df['Volume'].pct_change().abs() + 0.001)
        ).rolling(window=10).mean().fillna(0)

        # Clip extreme values
        df['Price_Impact'] = df['Price_Impact'].clip(-10, 10)

        return df

    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators.

        These indicators provide additional perspectives on price trends,
        momentum, and volatility beyond the basic indicators.
        """
        # ===== KELTNER CHANNELS =====
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        ema_20 = typical_price.ewm(span=20, adjust=False).mean()
        atr = self._calculate_atr(df, period=10)

        df['Keltner_Upper'] = ema_20 + (atr * 2)
        df['Keltner_Lower'] = ema_20 - (atr * 2)
        df['Keltner_Width'] = (df['Keltner_Upper'] - df['Keltner_Lower']) / ema_20
        df['Keltner_Position'] = (df['Close'] - df['Keltner_Lower']) / (df['Keltner_Upper'] - df['Keltner_Lower'] + 0.001)

        # ===== DONCHIAN CHANNELS =====
        df['Donchian_High'] = df['High'].rolling(window=20).max()
        df['Donchian_Low'] = df['Low'].rolling(window=20).min()
        df['Donchian_Mid'] = (df['Donchian_High'] + df['Donchian_Low']) / 2
        df['Donchian_Position'] = (df['Close'] - df['Donchian_Low']) / (df['Donchian_High'] - df['Donchian_Low'] + 0.001)

        # ===== CCI (Commodity Channel Index) =====
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['CCI'] = (tp - sma_tp) / (0.015 * mad + 0.001)
        df['CCI'] = df['CCI'].clip(-300, 300)  # Clip extreme values

        # ===== MFI (Money Flow Index) =====
        df['MFI'] = self._calculate_mfi(df, period=14)

        # ===== CMF (Chaikin Money Flow) =====
        df['CMF'] = self._calculate_cmf(df, period=20)

        # ===== FORCE INDEX =====
        df['Force_Index'] = df['Close'].diff() * df['Volume']
        df['Force_Index_EMA'] = df['Force_Index'].ewm(span=13, adjust=False).mean()
        # Normalize
        df['Force_Index_Norm'] = (
            df['Force_Index_EMA'] /
            df['Force_Index_EMA'].rolling(window=50).std()
        ).clip(-5, 5).fillna(0)

        # ===== TRIX (Triple Exponential Average) =====
        df['TRIX'] = self._calculate_trix(df['Close'], period=15)

        # ===== ULTIMATE OSCILLATOR =====
        df['Ultimate_Osc'] = self._calculate_ultimate_oscillator(df)

        # ===== CHOPPINESS INDEX (market trending vs ranging) =====
        df['Choppiness'] = self._calculate_choppiness(df, period=14)

        return df

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        # Get positive and negative money flow
        tp_diff = typical_price.diff()
        positive_flow = money_flow.where(tp_diff > 0, 0)
        negative_flow = money_flow.where(tp_diff < 0, 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1)))
        return mfi.fillna(50)

    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 0.001) * df['Volume']
        cmf = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
        return cmf.fillna(0)

    def _calculate_trix(self, prices: pd.Series, period: int = 15) -> pd.Series:
        """Calculate TRIX (Triple Exponential Average)"""
        ema1 = prices.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        trix = ema3.pct_change() * 100
        return trix.fillna(0)

    def _calculate_ultimate_oscillator(
        self,
        df: pd.DataFrame,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28
    ) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        bp = df['Close'] - pd.concat([df['Low'], df['Close'].shift(1)], axis=1).min(axis=1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)

        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()

        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        return uo.fillna(50)

    def _calculate_choppiness(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Choppiness Index.
        High values (>61.8) indicate ranging/choppy market.
        Low values (<38.2) indicate trending market.
        """
        atr_sum = self._calculate_atr(df, period=1).rolling(window=period).sum()
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()

        choppiness = 100 * np.log10(atr_sum / (high_max - low_min + 0.001)) / np.log10(period)
        return choppiness.clip(0, 100).fillna(50)

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price pattern features.

        These features help identify chart patterns and candlestick formations
        that may precede significant price movements.
        """
        # Higher High (current high > previous high)
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)

        # Lower Low (current low < previous low)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)

        # Inside Bar (high lower than prev, low higher than prev)
        df['Inside_Bar'] = (
            (df['High'] < df['High'].shift(1)) &
            (df['Low'] > df['Low'].shift(1))
        ).astype(int)

        # Outside Bar (high higher than prev, low lower than prev)
        df['Outside_Bar'] = (
            (df['High'] > df['High'].shift(1)) &
            (df['Low'] < df['Low'].shift(1))
        ).astype(int)

        # Gap Percentage (gap from previous close)
        df['Gap_Percentage'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100).fillna(0)
        df['Gap_Percentage'] = df['Gap_Percentage'].clip(-10, 10)

        # Gap Up / Gap Down flags
        df['Gap_Up'] = (df['Open'] > df['High'].shift(1)).astype(int)
        df['Gap_Down'] = (df['Open'] < df['Low'].shift(1)).astype(int)

        # Candle Body Size (relative to range)
        candle_range = df['High'] - df['Low']
        candle_body = (df['Close'] - df['Open']).abs()
        df['Body_Ratio'] = (candle_body / (candle_range + 0.001)).clip(0, 1)

        # Upper Shadow Ratio
        upper_shadow = df['High'] - pd.concat([df['Close'], df['Open']], axis=1).max(axis=1)
        df['Upper_Shadow_Ratio'] = (upper_shadow / (candle_range + 0.001)).clip(0, 1)

        # Lower Shadow Ratio
        lower_shadow = pd.concat([df['Close'], df['Open']], axis=1).min(axis=1) - df['Low']
        df['Lower_Shadow_Ratio'] = (lower_shadow / (candle_range + 0.001)).clip(0, 1)

        # Consecutive Up/Down days
        df['Up_Day'] = (df['Close'] > df['Close'].shift(1)).astype(int)
        df['Consecutive_Up'] = df['Up_Day'].groupby((df['Up_Day'] != df['Up_Day'].shift()).cumsum()).cumsum()
        df['Consecutive_Down'] = (1 - df['Up_Day']).groupby(((1 - df['Up_Day']) != (1 - df['Up_Day']).shift()).cumsum()).cumsum()

        # Distance from recent high/low
        df['Dist_From_20d_High'] = (df['Close'] - df['High'].rolling(20).max()) / df['Close']
        df['Dist_From_20d_Low'] = (df['Close'] - df['Low'].rolling(20).min()) / df['Close']

        return df

    def _add_cross_asset_features(self, df: pd.DataFrame, tasi_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add cross-asset and relative strength features.

        These features capture how the stock performs relative to the broader
        market (TASI index) and its sector.
        """
        # If TASI data is provided, calculate relative strength
        if tasi_data is not None and 'Close' in tasi_data.columns:
            # Align TASI data with stock data
            tasi_aligned = tasi_data['Close'].reindex(df.index, method='ffill')
            df['TASI_Return'] = tasi_aligned.pct_change().fillna(0)
            df['Relative_Strength_vs_TASI'] = (
                df['Daily_Return'] - df['TASI_Return']
            ).fillna(0)
            df['RS_Cumulative'] = (1 + df['Relative_Strength_vs_TASI']).cumprod() - 1
        else:
            # Use placeholder features (will be calculated later if TASI data available)
            df['TASI_Return'] = 0
            df['Relative_Strength_vs_TASI'] = 0
            df['RS_Cumulative'] = 0

        # Beta (20-day rolling) - volatility relative to self
        returns = df['Close'].pct_change().fillna(0)
        df['Rolling_Beta'] = returns.rolling(window=20).std() / returns.std()
        df['Rolling_Beta'] = df['Rolling_Beta'].fillna(1)

        # Alpha (excess return over expected)
        expected_return = returns.rolling(window=20).mean()
        df['Rolling_Alpha'] = returns - expected_return
        df['Rolling_Alpha'] = df['Rolling_Alpha'].fillna(0)

        return df

    def add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add macroeconomic features: Oil_Correlation and Market_Trend.

        Prerequisites:
            df must contain 'Oil_Close' and 'TASI_Close' columns
            (from fetch_stock_with_macro or manually merged)

        Features added:
            - Oil_Correlation: Rolling 30-day correlation between Stock Close and Oil price
            - Market_Trend: Rolling 14-day return of TASI index

        Args:
            df: DataFrame with stock data + Oil_Close + TASI_Close columns

        Returns:
            DataFrame with macro features added
        """
        df = df.copy()

        # ===== OIL CORRELATION (30-day rolling) =====
        if 'Oil_Close' in df.columns and df['Oil_Close'].notna().sum() > 30:
            # Calculate rolling 30-day correlation between Stock Close and Oil Close
            df['Oil_Correlation'] = (
                df['Close']
                .rolling(window=30, min_periods=15)
                .corr(df['Oil_Close'])
                .fillna(0)
            )
            # Clip extreme values
            df['Oil_Correlation'] = df['Oil_Correlation'].clip(-1, 1)
        else:
            # Placeholder if Oil data not available
            df['Oil_Correlation'] = 0
            print("[INFO] Oil_Close not available, Oil_Correlation set to 0")

        # ===== MARKET TREND (14-day rolling return of TASI) =====
        if 'TASI_Close' in df.columns and df['TASI_Close'].notna().sum() > 14:
            # Calculate 14-day rolling return of TASI
            # (current TASI close / TASI close 14 days ago) - 1
            df['Market_Trend'] = (
                df['TASI_Close'].pct_change(periods=14).fillna(0)
            )
            # Clip extreme values (e.g., max 50% move in 14 days)
            df['Market_Trend'] = df['Market_Trend'].clip(-0.5, 0.5)
        else:
            # Placeholder if TASI data not available
            df['Market_Trend'] = 0
            print("[INFO] TASI_Close not available, Market_Trend set to 0")

        # ===== ADDITIONAL MACRO FEATURES =====

        # Oil Momentum (5-day change in oil price)
        if 'Oil_Close' in df.columns and df['Oil_Close'].notna().sum() > 5:
            df['Oil_Momentum'] = df['Oil_Close'].pct_change(periods=5).fillna(0).clip(-0.3, 0.3)
        else:
            df['Oil_Momentum'] = 0

        # Market Volatility (14-day rolling std of TASI returns)
        if 'TASI_Close' in df.columns and df['TASI_Close'].notna().sum() > 14:
            tasi_returns = df['TASI_Close'].pct_change().fillna(0)
            df['Market_Volatility'] = (
                tasi_returns.rolling(window=14, min_periods=7).std().fillna(0)
            )
            # Annualize for interpretability
            df['Market_Volatility'] = df['Market_Volatility'] * np.sqrt(252)
        else:
            df['Market_Volatility'] = 0

        # Stock-Market Beta (rolling 30-day covariance / market variance)
        if 'TASI_Close' in df.columns and df['TASI_Close'].notna().sum() > 30:
            stock_returns = df['Close'].pct_change().fillna(0)
            market_returns = df['TASI_Close'].pct_change().fillna(0)

            # Rolling covariance
            rolling_cov = stock_returns.rolling(window=30, min_periods=15).cov(market_returns)
            # Rolling market variance
            rolling_var = market_returns.rolling(window=30, min_periods=15).var()

            df['Stock_Market_Beta'] = (rolling_cov / (rolling_var + 1e-8)).fillna(1).clip(-3, 3)
        else:
            df['Stock_Market_Beta'] = 1

        return df

    def get_macro_feature_list(self) -> List[str]:
        """
        Get the list of macroeconomic features.

        Returns:
            List of macro feature names
        """
        return [
            'Oil_Correlation',     # 30-day rolling correlation with Brent Oil
            'Market_Trend',        # 14-day rolling return of TASI
            'Oil_Momentum',        # 5-day change in oil price
            'Market_Volatility',   # 14-day rolling std of TASI returns (annualized)
            'Stock_Market_Beta',   # 30-day rolling beta vs TASI
        ]

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features for better model performance.

        These features capture distributional characteristics of returns
        that may be predictive of future volatility and direction.
        """
        returns = df['Close'].pct_change().fillna(0)

        # Skewness (20-day rolling)
        df['Return_Skewness'] = returns.rolling(window=20).skew().fillna(0)

        # Kurtosis (20-day rolling)
        df['Return_Kurtosis'] = returns.rolling(window=20).kurt().fillna(0)
        df['Return_Kurtosis'] = df['Return_Kurtosis'].clip(-10, 10)

        # Z-score of current price vs 20-day mean
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['Price_Zscore'] = ((df['Close'] - rolling_mean) / (rolling_std + 0.001)).clip(-3, 3).fillna(0)

        # Percentile rank of current close in 50-day range
        df['Price_Percentile'] = df['Close'].rolling(window=50).apply(
            lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 0.001) if len(x) > 0 else 0.5,
            raw=True
        ).fillna(0.5)

        # Historical Volatility (annualized)
        df['HV_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['HV_20'] = df['HV_20'].fillna(df['HV_20'].mean())

        # Volatility Regime (current vol vs average)
        avg_vol = df['HV_20'].rolling(window=60).mean()
        df['Vol_Regime'] = (df['HV_20'] / (avg_vol + 0.001)).clip(0.5, 3).fillna(1)

        # Average True Range Percentage
        df['ATR_Percent'] = (df['ATR'] / df['Close'] * 100).fillna(0)

        return df

    def get_advanced_feature_list(self, include_macro: bool = True) -> List[str]:
        """
        Get the list of all features for advanced LSTM model.

        Args:
            include_macro: Whether to include macroeconomic features

        Returns:
            List of 40+ feature names (including macro features)
        """
        features = [
            # Basic OHLCV
            'Close', 'Volume', 'High', 'Low',
            # Moving Averages
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            # Momentum Indicators
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            # Volatility
            'ATR', 'BB_Width', 'Volatility', 'HV_20',
            # Volume Indicators
            'OBV', 'Volume_MA', 'Volume_Ratio', 'Relative_Volume',
            # Oscillators
            'Stoch_K', 'Williams_R', 'ADX', 'CCI', 'MFI', 'Ultimate_Osc',
            # Microstructure
            'Volume_Momentum', 'Price_Volume_Corr', 'Amihud_Illiquidity', 'Volume_Volatility',
            # Channels
            'Keltner_Position', 'Donchian_Position',
            # Pattern Features
            'Gap_Percentage', 'Body_Ratio', 'Dist_From_20d_High', 'Dist_From_20d_Low',
            # Statistical
            'Price_Zscore', 'Price_Percentile', 'Return_Skewness', 'Vol_Regime',
            # Trend
            'ROC', 'Momentum', 'Daily_Return', 'Choppiness', 'TRIX',
            # Cross-Asset
            'Rolling_Beta', 'Rolling_Alpha'
        ]

        if include_macro:
            features.extend(self.get_macro_feature_list())

        return features

    def prepare_lstm_data(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        sequence_length: int = None,
        train_split: float = None,
        val_split: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare data for LSTM model training with proper train/val/test split
        FIXED: Scaler fits ONLY on training data to prevent data leakage

        Args:
            df: DataFrame with technical indicators
            features: List of feature columns to use
            sequence_length: Number of time steps to look back
            train_split: Ratio of training data
            val_split: Ratio of validation data (from remaining after train)

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, scaler
        """
        if features is None:
            features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'OBV']
        if sequence_length is None:
            sequence_length = ADVANCED_LSTM_CONFIG['sequence_length']
        if train_split is None:
            train_split = ADVANCED_LSTM_CONFIG['train_split']

        # Filter available features
        available_features = [f for f in features if f in df.columns]

        # Extract feature data
        data = df[available_features].values

        # CRITICAL FIX: Split data BEFORE scaling
        train_size = int(len(data) * train_split)
        val_size = int(len(data) * val_split)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        # Fit scaler ONLY on training data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = self.scaler.fit_transform(train_data)
        self.scaler_fitted = True

        # Transform validation and test data using train scaler
        scaled_val = self.scaler.transform(val_data)
        scaled_test = self.scaler.transform(test_data)

        # Also fit a separate scaler for Close price (for inverse transform later)
        close_idx = available_features.index('Close') if 'Close' in available_features else 0
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler.fit(train_data[:, close_idx:close_idx+1])

        # Create sequences for each split
        def create_sequences(scaled_data, seq_len):
            X, y = [], []
            for i in range(seq_len, len(scaled_data)):
                X.append(scaled_data[i-seq_len:i])
                y.append(scaled_data[i, 0])  # Target: next day's Close price
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(scaled_train, sequence_length)
        X_val, y_val = create_sequences(scaled_val, sequence_length) if len(scaled_val) > sequence_length else (np.array([]), np.array([]))
        X_test, y_test = create_sequences(scaled_test, sequence_length) if len(scaled_test) > sequence_length else (np.array([]), np.array([]))

        return X_train, X_val, X_test, y_train, y_val, y_test, self.scaler

    def inverse_transform_price(self, scaled_price: np.ndarray) -> np.ndarray:
        """Convert scaled price back to original scale"""
        if not self.scaler_fitted:
            raise ValueError("Scaler not fitted. Call prepare_lstm_data first.")

        dummy = np.zeros((len(scaled_price), self.scaler.n_features_in_))
        dummy[:, 0] = scaled_price.flatten()
        unscaled = self.scaler.inverse_transform(dummy)
        return unscaled[:, 0]

    def prepare_lstm_data_with_split(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        sequence_length: int = None,
        train_split: float = None,
        val_split: float = None
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for LSTM model training with proper train/val/test split
        Returns a dictionary for easier access.

        Args:
            df: DataFrame with technical indicators
            features: List of feature columns to use
            sequence_length: Number of time steps to look back
            train_split: Ratio of training data
            val_split: Ratio of validation data

        Returns:
            Dictionary with X_train, X_val, X_test, y_train, y_val, y_test, scaler
        """
        if train_split is None:
            train_split = ADVANCED_LSTM_CONFIG.get('train_split', 0.7)
        if val_split is None:
            val_split = ADVANCED_LSTM_CONFIG.get('val_split', 0.15)

        X_train, X_val, X_test, y_train, y_val, y_test, scaler = self.prepare_lstm_data(
            df,
            features=features,
            sequence_length=sequence_length,
            train_split=train_split,
            val_split=val_split
        )

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler
        }

    def get_latest_sequence(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        sequence_length: int = None
    ) -> np.ndarray:
        """
        Get the latest sequence for prediction

        Args:
            df: DataFrame with technical indicators
            features: List of feature columns
            sequence_length: Sequence length

        Returns:
            Scaled sequence ready for model input
        """
        if features is None:
            features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'OBV']
        if sequence_length is None:
            sequence_length = ADVANCED_LSTM_CONFIG['sequence_length']

        available_features = [f for f in features if f in df.columns]
        data = df[available_features].tail(sequence_length).values

        if not self.scaler_fitted:
            raise ValueError("Scaler not fitted. Call prepare_lstm_data first.")

        scaled_data = self.scaler.transform(data)
        return scaled_data.reshape(1, sequence_length, len(available_features))

    def save_scaler(self, filepath: str):
        """Save the fitted scaler to disk"""
        if self.scaler_fitted:
            joblib.dump({
                'scaler': self.scaler,
                'price_scaler': self.price_scaler
            }, filepath)

    def load_scaler(self, filepath: str):
        """Load a fitted scaler from disk"""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.scaler = data['scaler']
            self.price_scaler = data['price_scaler']
            self.scaler_fitted = True
            return True
        return False


def preprocess_stock_data(
    df: pd.DataFrame,
    include_advanced: bool = True,
    include_macro: bool = True
) -> pd.DataFrame:
    """
    Convenience function to preprocess stock data

    Args:
        df: Raw stock DataFrame (optionally with Oil_Close, TASI_Close columns)
        include_advanced: Whether to include advanced features (35+ features)
        include_macro: Whether to include macroeconomic features

    Returns:
        Preprocessed DataFrame with technical indicators
    """
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_data(df)
    df = preprocessor.add_technical_indicators(df)

    if include_advanced:
        df = preprocessor.add_advanced_features(df)

    if include_macro and ('Oil_Close' in df.columns or 'TASI_Close' in df.columns):
        df = preprocessor.add_macro_features(df)

    return df


def preprocess_for_advanced_lstm(
    df: pd.DataFrame,
    tasi_data: pd.DataFrame = None,
    include_macro: bool = True
) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Preprocess stock data specifically for the Advanced LSTM model.

    This function applies all preprocessing steps including advanced features
    and macroeconomic features optimized for the high-accuracy AdvancedStockLSTM model.

    Args:
        df: Raw stock DataFrame (optionally with Oil_Close, TASI_Close columns)
        tasi_data: Optional TASI index data for relative strength features
        include_macro: Whether to include macroeconomic features (default True)

    Returns:
        Tuple of (preprocessed DataFrame, DataPreprocessor instance)
    """
    preprocessor = DataPreprocessor()

    # Clean data
    df = preprocessor.clean_data(df)

    # Add basic technical indicators
    df = preprocessor.add_technical_indicators(df)

    # Add advanced features
    df = preprocessor.add_advanced_features(df)

    # Add cross-asset features if TASI data provided
    if tasi_data is not None:
        df = preprocessor._add_cross_asset_features(df, tasi_data)

    # Add macroeconomic features if data is available
    if include_macro and ('Oil_Close' in df.columns or 'TASI_Close' in df.columns):
        df = preprocessor.add_macro_features(df)

    return df, preprocessor


if __name__ == "__main__":
    from data_loader import SaudiStockDataLoader

    print("Testing Data Preprocessor...")
    print("=" * 60)

    loader = SaudiStockDataLoader()
    raw_data = loader.fetch_stock_data("2222", period="1y")

    print(f"Raw data shape: {raw_data.shape}")

    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean_data(raw_data)
    processed_data = preprocessor.add_technical_indicators(clean_data)

    print(f"Processed data shape: {processed_data.shape}")
    print(f"\nColumns: {list(processed_data.columns)}")
    print(f"\nLast 5 rows:")
    print(processed_data.tail())

    # Test LSTM preparation with proper splits
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocessor.prepare_lstm_data(processed_data)
    print(f"\nLSTM data shapes (with proper train/val/test split):")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  y_test: {y_test.shape}")
