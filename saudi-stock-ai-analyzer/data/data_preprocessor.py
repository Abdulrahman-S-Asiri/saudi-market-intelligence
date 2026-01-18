"""
Data preprocessing and cleaning module for Saudi Stock AI Analyzer
Includes technical indicators calculation and LSTM data preparation
FIXED: Data leakage issue - scaler now fits only on training data
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import INDICATORS, LSTM_CONFIG


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
        col_mapping = {}
        for col in df.columns:
            col_str = str(col)
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

        # Ensure numeric types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

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
            sequence_length = LSTM_CONFIG['sequence_length']
        if train_split is None:
            train_split = LSTM_CONFIG['train_split']

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
            train_split = LSTM_CONFIG.get('train_split', 0.7)
        if val_split is None:
            val_split = LSTM_CONFIG.get('val_split', 0.15)

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
            sequence_length = LSTM_CONFIG['sequence_length']

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


def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to preprocess stock data

    Args:
        df: Raw stock DataFrame

    Returns:
        Preprocessed DataFrame with technical indicators
    """
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_data(df)
    df = preprocessor.add_technical_indicators(df)
    return df


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
