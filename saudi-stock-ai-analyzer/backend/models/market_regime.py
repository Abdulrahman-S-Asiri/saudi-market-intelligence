# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
Market Regime Detection using Hidden Markov Models (HMM)

This module implements a regime detection system that classifies the overall
market state into Bull, Bear, or Sideways conditions using the TASI Index.

Key Features:
    - Uses Gaussian HMM with 3 hidden states
    - Trained on daily returns and rolling volatility
    - Provides regime labels and transition probabilities
    - Helps users decide IF they should be in the market

Usage:
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(tasi_data)
    # Returns: {'regime': 'Bull', 'confidence': 0.85, 'warning': None}
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging
import pickle
import os

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not installed. Run: pip install hmmlearn")

logger = logging.getLogger(__name__)


# ============================================================
# REGIME CONFIGURATION
# ============================================================

REGIME_CONFIG = {
    'n_states': 3,              # Bull, Bear, Sideways
    'n_iter': 100,              # HMM training iterations
    'volatility_window': 20,    # Rolling window for volatility
    'min_data_points': 60,      # Minimum days of data required
    'model_cache_path': 'models/saved/regime_model.pkl',
}

# State labels (assigned after training based on mean returns)
REGIME_LABELS = {
    'bull': {'name': 'Bull Market', 'emoji': '🟢', 'color': '#00ffa3'},
    'bear': {'name': 'Bear Market', 'emoji': '🔴', 'color': '#ff4757'},
    'sideways': {'name': 'Sideways', 'emoji': '🟡', 'color': '#ffc107'},
}


class MarketRegimeDetector:
    """
    Hidden Markov Model-based Market Regime Detector.

    Classifies market conditions into 3 regimes:
        1. Bull Market (🟢): High positive returns, low volatility
        2. Bear Market (🔴): Negative returns, high volatility
        3. Sideways (🟡): Low returns, moderate volatility

    The model is trained on:
        - Daily returns of TASI Index
        - Rolling volatility (20-day standard deviation)

    Attributes:
        model: Trained GaussianHMM model
        state_labels: Mapping of HMM states to regime labels
        is_trained: Whether the model has been trained
    """

    def __init__(self, n_states: int = 3, random_state: int = 42):
        """
        Initialize the Market Regime Detector.

        Args:
            n_states: Number of hidden states (default: 3 for Bull/Bear/Sideways)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self.state_labels: Dict[int, str] = {}
        self.state_means: Dict[int, float] = {}
        self.is_trained = False
        self.last_training_date: Optional[datetime] = None

        if HMM_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("hmmlearn not available. Using fallback regime detection.")

    def _initialize_model(self):
        """Initialize the Gaussian HMM model."""
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=REGIME_CONFIG['n_iter'],
            random_state=self.random_state,
            verbose=False
        )

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM training/prediction.

        Features:
            1. Daily Returns: (Close_t - Close_{t-1}) / Close_{t-1}
            2. Rolling Volatility: 20-day rolling std of returns

        Args:
            df: DataFrame with 'Close' column (TASI Index data)

        Returns:
            Feature matrix of shape (n_samples, 2)
        """
        # Ensure we have a Close column
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")

        # Calculate daily returns
        returns = df['Close'].pct_change()

        # Calculate rolling volatility (20-day window)
        volatility = returns.rolling(
            window=REGIME_CONFIG['volatility_window']
        ).std()

        # Combine features
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility
        }).dropna()

        return features.values

    def _assign_state_labels(self, features: np.ndarray):
        """
        Assign semantic labels to HMM states based on their characteristics.

        Logic:
            - State with highest mean return → Bull
            - State with lowest mean return → Bear
            - Remaining state → Sideways
        """
        # Predict states for all data points
        states = self.model.predict(features)

        # Calculate mean return for each state
        state_returns = {}
        for state in range(self.n_states):
            state_mask = states == state
            if np.sum(state_mask) > 0:
                state_returns[state] = features[state_mask, 0].mean()
            else:
                state_returns[state] = 0

        # Sort states by mean return
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])

        # Assign labels: lowest return = Bear, highest = Bull, middle = Sideways
        if self.n_states == 3:
            self.state_labels = {
                sorted_states[0][0]: 'bear',      # Lowest return
                sorted_states[1][0]: 'sideways',  # Middle return
                sorted_states[2][0]: 'bull',      # Highest return
            }
        elif self.n_states == 2:
            self.state_labels = {
                sorted_states[0][0]: 'bear',
                sorted_states[1][0]: 'bull',
            }

        # Store mean returns for each state
        self.state_means = state_returns

        logger.info(f"State labels assigned: {self.state_labels}")
        logger.info(f"State mean returns: {state_returns}")

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the HMM on TASI Index data.

        Args:
            df: DataFrame with 'Close' column (at least 60 days of data)

        Returns:
            Training info dictionary
        """
        if not HMM_AVAILABLE:
            return {'status': 'error', 'message': 'hmmlearn not installed'}

        # Prepare features
        features = self._prepare_features(df)

        if len(features) < REGIME_CONFIG['min_data_points']:
            return {
                'status': 'error',
                'message': f"Need at least {REGIME_CONFIG['min_data_points']} data points"
            }

        # Train the model
        logger.info(f"Training HMM on {len(features)} data points...")
        self.model.fit(features)

        # Assign semantic labels to states
        self._assign_state_labels(features)

        self.is_trained = True
        self.last_training_date = datetime.now()

        # Calculate training statistics
        states = self.model.predict(features)
        state_counts = {self.state_labels[s]: np.sum(states == s) for s in range(self.n_states)}

        return {
            'status': 'success',
            'n_samples': len(features),
            'state_distribution': state_counts,
            'state_means': {self.state_labels[k]: round(v * 100, 3) for k, v in self.state_means.items()},
            'training_date': self.last_training_date.isoformat()
        }

    def detect_regime(self, df: pd.DataFrame) -> Dict:
        """
        Detect the current market regime.

        If not trained, will automatically train on the provided data.

        Args:
            df: DataFrame with 'Close' column (TASI Index data)

        Returns:
            Dictionary with:
            - regime: Current regime label ('bull', 'bear', 'sideways')
            - regime_name: Human-readable name
            - emoji: Regime emoji
            - confidence: Probability of current state
            - warning: Warning message if bearish
            - state_probabilities: Probabilities for all states
        """
        # Fallback if HMM not available
        if not HMM_AVAILABLE:
            return self._fallback_detection(df)

        # Auto-train if not trained
        if not self.is_trained:
            train_result = self.train(df)
            if train_result['status'] == 'error':
                return self._fallback_detection(df)

        # Prepare features
        features = self._prepare_features(df)

        if len(features) == 0:
            return self._fallback_detection(df)

        # Get the last data point for current regime
        latest_features = features[-1:, :]

        # Predict current state
        current_state = self.model.predict(latest_features)[0]

        # Get state probabilities
        state_probs = self.model.predict_proba(latest_features)[0]

        # Get regime label
        regime = self.state_labels.get(current_state, 'sideways')
        regime_info = REGIME_LABELS[regime]
        confidence = float(state_probs[current_state])

        # Generate warning for bearish regime
        warning = None
        if regime == 'bear' and confidence > 0.5:
            warning = "⚠️ Caution: Market is in downtrend. Consider reducing exposure."
        elif regime == 'bear':
            warning = "⚠️ Market showing bearish signals. Trade with caution."

        # Calculate regime probabilities with labels
        regime_probabilities = {
            self.state_labels[s]: round(float(state_probs[s]) * 100, 1)
            for s in range(self.n_states)
        }

        return {
            'regime': regime,
            'regime_name': regime_info['name'],
            'emoji': regime_info['emoji'],
            'color': regime_info['color'],
            'confidence': round(confidence * 100, 1),
            'warning': warning,
            'regime_probabilities': regime_probabilities,
            'detection_method': 'hmm',
            'timestamp': datetime.now().isoformat()
        }

    def _fallback_detection(self, df: pd.DataFrame) -> Dict:
        """
        Fallback regime detection using simple rules when HMM is not available.

        Rules:
            - 20-day return > 2% and RSI < 70 → Bull
            - 20-day return < -2% or RSI > 80 → Bear
            - Otherwise → Sideways
        """
        if 'Close' not in df.columns or len(df) < 20:
            return {
                'regime': 'sideways',
                'regime_name': 'Sideways',
                'emoji': '🟡',
                'color': '#ffc107',
                'confidence': 50.0,
                'warning': None,
                'regime_probabilities': {'bull': 33.3, 'bear': 33.3, 'sideways': 33.4},
                'detection_method': 'fallback',
                'timestamp': datetime.now().isoformat()
            }

        # Calculate indicators
        returns_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
        returns_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100

        # Simple RSI calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # Volatility
        volatility = df['Close'].pct_change().rolling(20).std().iloc[-1] * 100

        # Determine regime
        if returns_20d > 3 and rsi < 70:
            regime = 'bull'
            confidence = min(70 + returns_20d * 2, 95)
        elif returns_20d < -3 or (rsi > 75 and returns_5d < -1):
            regime = 'bear'
            confidence = min(70 + abs(returns_20d) * 2, 95)
        else:
            regime = 'sideways'
            confidence = 60

        regime_info = REGIME_LABELS[regime]

        warning = None
        if regime == 'bear':
            warning = "⚠️ Caution: Market is in downtrend. Consider reducing exposure."

        return {
            'regime': regime,
            'regime_name': regime_info['name'],
            'emoji': regime_info['emoji'],
            'color': regime_info['color'],
            'confidence': round(confidence, 1),
            'warning': warning,
            'regime_probabilities': {
                'bull': round(33.3 + (returns_20d * 2 if returns_20d > 0 else 0), 1),
                'bear': round(33.3 + (abs(returns_20d) * 2 if returns_20d < 0 else 0), 1),
                'sideways': 33.4
            },
            'indicators': {
                'return_20d': round(returns_20d, 2),
                'return_5d': round(returns_5d, 2),
                'rsi': round(rsi, 1),
                'volatility': round(volatility, 2)
            },
            'detection_method': 'fallback',
            'timestamp': datetime.now().isoformat()
        }

    def get_regime_history(self, df: pd.DataFrame, lookback: int = 60) -> List[Dict]:
        """
        Get historical regime states for visualization.

        Args:
            df: DataFrame with 'Close' column
            lookback: Number of days to analyze

        Returns:
            List of regime states with dates
        """
        if not self.is_trained:
            self.train(df)

        if not HMM_AVAILABLE or not self.is_trained:
            return []

        features = self._prepare_features(df)

        # Limit to lookback period
        features = features[-lookback:]

        # Predict all states
        states = self.model.predict(features)
        probs = self.model.predict_proba(features)

        # Get dates (accounting for feature preparation dropping some rows)
        dates = df.index[-len(states):]

        history = []
        for i, (date, state) in enumerate(zip(dates, states)):
            regime = self.state_labels.get(state, 'sideways')
            history.append({
                'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                'regime': regime,
                'confidence': round(float(probs[i, state]) * 100, 1)
            })

        return history

    def save_model(self, path: str = None):
        """Save trained model to disk."""
        if not self.is_trained:
            logger.warning("Model not trained. Nothing to save.")
            return

        path = path or REGIME_CONFIG['model_cache_path']
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'state_labels': self.state_labels,
                'state_means': self.state_means,
                'last_training_date': self.last_training_date
            }, f)

        logger.info(f"Regime model saved to {path}")

    def load_model(self, path: str = None) -> bool:
        """Load trained model from disk."""
        path = path or REGIME_CONFIG['model_cache_path']

        if not os.path.exists(path):
            return False

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.model = data['model']
            self.state_labels = data['state_labels']
            self.state_means = data['state_means']
            self.last_training_date = data['last_training_date']
            self.is_trained = True

            logger.info(f"Regime model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load regime model: {e}")
            return False


# ============================================================
# SINGLETON INSTANCE
# ============================================================

# Global regime detector instance
_regime_detector: Optional[MarketRegimeDetector] = None


def get_regime_detector() -> MarketRegimeDetector:
    """Get or create the global regime detector instance."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector()
    return _regime_detector


def detect_market_regime(tasi_data: pd.DataFrame) -> Dict:
    """
    Convenience function to detect market regime.

    Args:
        tasi_data: DataFrame with TASI Index data

    Returns:
        Regime detection result
    """
    detector = get_regime_detector()
    return detector.detect_regime(tasi_data)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    import yfinance as yf

    print("=" * 60)
    print("Market Regime Detector Test")
    print("=" * 60)

    # Fetch TASI data
    print("\nFetching TASI Index data...")
    tasi = yf.download("^TASI.SR", period="1y", progress=False)
    print(f"Downloaded {len(tasi)} days of data")

    # Create detector
    detector = MarketRegimeDetector()

    # Train
    print("\nTraining HMM...")
    train_result = detector.train(tasi)
    print(f"Training result: {train_result}")

    # Detect current regime
    print("\nDetecting current regime...")
    regime = detector.detect_regime(tasi)
    print(f"Current Regime: {regime['regime_name']} {regime['emoji']}")
    print(f"Confidence: {regime['confidence']}%")
    print(f"Probabilities: {regime['regime_probabilities']}")
    if regime['warning']:
        print(f"Warning: {regime['warning']}")

    print("\n" + "=" * 60)
    print("Test Complete!")
