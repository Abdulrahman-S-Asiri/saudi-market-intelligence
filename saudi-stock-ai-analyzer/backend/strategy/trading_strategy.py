# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
Enhanced trading strategy with adaptive thresholds and regime detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import RSI_OVERSOLD, RSI_OVERBOUGHT, STRATEGY_CONFIG, REGIME_CONFIG


class Signal(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    action: str
    confidence: float
    price: float
    reasons: List[str]
    technical_score: float
    lstm_score: float
    regime: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    indicators: Optional[Dict] = None
    market_regime: Optional[str] = None
    trend_strength: Optional[float] = None


class AdaptiveThresholds:
    """Adaptive threshold calculator based on volatility"""

    def __init__(self):
        self.base_rsi_oversold = RSI_OVERSOLD
        self.base_rsi_overbought = RSI_OVERBOUGHT

    def calculate(self, df: pd.DataFrame) -> Dict:
        """Calculate adaptive thresholds based on current volatility"""
        if 'Volatility' not in df.columns or len(df) < 20:
            return {
                'rsi_oversold': self.base_rsi_oversold,
                'rsi_overbought': self.base_rsi_overbought,
                'volume_threshold': 1.5,
                'macd_threshold': 0
            }

        current_vol = df['Volatility'].iloc[-1] if not pd.isna(df['Volatility'].iloc[-1]) else 0.02
        avg_vol = df['Volatility'].mean()
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        rsi_adjustment = min(max(vol_ratio - 1, -5), 5) * 3
        rsi_oversold = max(20, self.base_rsi_oversold - rsi_adjustment)
        rsi_overbought = min(80, self.base_rsi_overbought + rsi_adjustment)

        volume_threshold = 1.2 + (0.5 / vol_ratio) if vol_ratio > 0 else 1.5
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
        macd_threshold = atr * 0.1

        return {
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'volume_threshold': volume_threshold,
            'macd_threshold': macd_threshold,
            'volatility_ratio': vol_ratio
        }


class RegimeDetector:
    """Detect market regime (trending vs ranging)"""

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        if len(df) < 50:
            return MarketRegime.SIDEWAYS

        close = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else close
        sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else close
        adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 25

        above_20 = close > sma_20
        above_50 = close > sma_50
        sma_20_above_50 = sma_20 > sma_50

        trend_score = sum([above_20, above_50, sma_20_above_50])

        if len(df) >= 5:
            momentum = (close - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
        else:
            momentum = 0

        if adx > 30:
            if trend_score >= 2 and momentum > 2:
                return MarketRegime.STRONG_UPTREND
            elif trend_score >= 2:
                return MarketRegime.UPTREND
            elif trend_score <= 1 and momentum < -2:
                return MarketRegime.STRONG_DOWNTREND
            elif trend_score <= 1:
                return MarketRegime.DOWNTREND

        return MarketRegime.SIDEWAYS

    def get_regime_adjustments(self, regime: MarketRegime) -> Dict:
        """Get signal adjustments based on regime"""
        adjustments = {
            MarketRegime.STRONG_UPTREND: {'buy_bias': 0.2, 'sell_threshold': 0.7, 'confidence_boost': 0.1},
            MarketRegime.UPTREND: {'buy_bias': 0.1, 'sell_threshold': 0.6, 'confidence_boost': 0.05},
            MarketRegime.SIDEWAYS: {'buy_bias': 0, 'sell_threshold': 0.5, 'confidence_boost': 0},
            MarketRegime.DOWNTREND: {'buy_bias': -0.1, 'sell_threshold': 0.4, 'confidence_boost': -0.05},
            MarketRegime.STRONG_DOWNTREND: {'buy_bias': -0.2, 'sell_threshold': 0.3, 'confidence_boost': -0.1}
        }
        return adjustments.get(regime, adjustments[MarketRegime.SIDEWAYS])


class SignalHysteresis:
    """Prevent signal whipsaws with hysteresis"""

    def __init__(self, cooldown_periods: int = 3):
        self.cooldown_periods = cooldown_periods
        self.last_signal = None
        self.last_signal_period = 0
        self.current_period = 0

    def filter_signal(self, signal: str, confidence: float) -> Tuple[str, float]:
        """Apply hysteresis to prevent rapid signal changes"""
        self.current_period += 1
        periods_since_last = self.current_period - self.last_signal_period

        if self.last_signal and periods_since_last < self.cooldown_periods:
            if signal != self.last_signal and signal != 'HOLD':
                required_confidence = 70 + (self.cooldown_periods - periods_since_last) * 5
                if confidence < required_confidence:
                    return 'HOLD', confidence * 0.8

        if signal != 'HOLD':
            self.last_signal = signal
            self.last_signal_period = self.current_period

        return signal, confidence


class MarketRegimeFilter:
    """
    Filter signals based on market regime to avoid counter-trend trades.
    Only allow BUY in uptrends/sideways, SELL in downtrends/sideways.
    """

    def __init__(self):
        self.trend_strength_threshold = REGIME_CONFIG.get('trend_strength_threshold', 0.02)
        self.adx_trending_threshold = REGIME_CONFIG.get('adx_trending_threshold', 25)

    def filter_signal(self, signal: str, confidence: float, regime: MarketRegime,
                      trend_strength: float = 0) -> Tuple[str, float, List[str]]:
        """
        Filter signal based on market regime.

        Returns:
            Tuple of (filtered_signal, adjusted_confidence, filter_reasons)
        """
        reasons = []

        # Strong uptrend - block SELL signals
        if regime == MarketRegime.STRONG_UPTREND:
            if signal == 'SELL':
                reasons.append("SELL blocked in strong uptrend")
                return 'HOLD', confidence * 0.5, reasons
            elif signal == 'BUY':
                confidence *= 1.1  # Boost confidence for trend-following
                reasons.append("BUY boosted in strong uptrend")

        # Strong downtrend - block BUY signals
        elif regime == MarketRegime.STRONG_DOWNTREND:
            if signal == 'BUY':
                reasons.append("BUY blocked in strong downtrend")
                return 'HOLD', confidence * 0.5, reasons
            elif signal == 'SELL':
                confidence *= 1.1
                reasons.append("SELL boosted in strong downtrend")

        # Regular uptrend - reduce SELL confidence
        elif regime == MarketRegime.UPTREND:
            if signal == 'SELL':
                confidence *= 0.7
                reasons.append("SELL confidence reduced in uptrend")
            elif signal == 'BUY':
                confidence *= 1.05

        # Regular downtrend - reduce BUY confidence
        elif regime == MarketRegime.DOWNTREND:
            if signal == 'BUY':
                confidence *= 0.7
                reasons.append("BUY confidence reduced in downtrend")
            elif signal == 'SELL':
                confidence *= 1.05

        # Sideways - slight penalty for both directions (mean reversion environment)
        else:  # SIDEWAYS
            # No adjustment needed - good for both BUY and SELL
            pass

        return signal, min(confidence, 100), reasons


class MultiIndicatorConfirmation:
    """
    Require multiple indicators to agree before taking a trade.
    Improves win rate by filtering out conflicting signals.
    """

    def __init__(self, min_confirming: int = 3):
        self.min_confirming = min_confirming

    def check_confirmation(self, df: pd.DataFrame, signal: str) -> Tuple[int, int, List[str]]:
        """
        Check how many indicators confirm the proposed signal.

        Args:
            df: DataFrame with indicator data
            signal: Proposed signal ('BUY' or 'SELL')

        Returns:
            Tuple of (confirming_count, total_indicators, confirmation_reasons)
        """
        if len(df) < 2 or signal == 'HOLD':
            return 0, 0, []

        current = df.iloc[-1]
        previous = df.iloc[-2]
        confirmations = 0
        total = 0
        reasons = []

        # 1. RSI Confirmation
        if 'RSI' in current and not pd.isna(current['RSI']):
            total += 1
            rsi = current['RSI']
            if signal == 'BUY' and rsi < 40:  # Oversold zone
                confirmations += 1
                reasons.append(f"RSI confirms BUY ({rsi:.1f} < 40)")
            elif signal == 'SELL' and rsi > 60:  # Overbought zone
                confirmations += 1
                reasons.append(f"RSI confirms SELL ({rsi:.1f} > 60)")

        # 2. MACD Confirmation
        if 'MACD' in current and 'MACD_Signal' in current:
            if not pd.isna(current['MACD']) and not pd.isna(current['MACD_Signal']):
                total += 1
                macd_bullish = current['MACD'] > current['MACD_Signal']
                if signal == 'BUY' and macd_bullish:
                    confirmations += 1
                    reasons.append("MACD confirms BUY (above signal)")
                elif signal == 'SELL' and not macd_bullish:
                    confirmations += 1
                    reasons.append("MACD confirms SELL (below signal)")

        # 3. Moving Average Alignment
        if 'SMA_20' in current and 'SMA_50' in current:
            if not pd.isna(current['SMA_20']) and not pd.isna(current['SMA_50']):
                total += 1
                close = current['Close']
                sma_20 = current['SMA_20']
                sma_50 = current['SMA_50']
                if signal == 'BUY' and close > sma_20 and sma_20 > sma_50:
                    confirmations += 1
                    reasons.append("MA alignment confirms BUY")
                elif signal == 'SELL' and close < sma_20 and sma_20 < sma_50:
                    confirmations += 1
                    reasons.append("MA alignment confirms SELL")

        # 4. Stochastic Confirmation
        stoch_col = 'Stoch_K' if 'Stoch_K' in current else 'Stochastic_K' if 'Stochastic_K' in current else None
        if stoch_col and not pd.isna(current[stoch_col]):
            total += 1
            stoch = current[stoch_col]
            if signal == 'BUY' and stoch < 30:  # Oversold
                confirmations += 1
                reasons.append(f"Stochastic confirms BUY ({stoch:.1f} < 30)")
            elif signal == 'SELL' and stoch > 70:  # Overbought
                confirmations += 1
                reasons.append(f"Stochastic confirms SELL ({stoch:.1f} > 70)")

        # 5. Volume Confirmation
        if 'Volume_Ratio' in current and not pd.isna(current['Volume_Ratio']):
            total += 1
            vol_ratio = current['Volume_Ratio']
            if vol_ratio > 1.2:  # Above average volume confirms any move
                confirmations += 1
                reasons.append(f"Volume confirms signal ({vol_ratio:.1f}x avg)")
        elif 'Volume' in current and 'Volume' in df.columns:
            total += 1
            avg_volume = df['Volume'].tail(20).mean()
            if current['Volume'] > avg_volume * 1.2:
                confirmations += 1
                reasons.append("Above average volume confirms")

        return confirmations, total, reasons

    def apply_confirmation_filter(self, signal: str, confidence: float,
                                   confirmations: int, total: int) -> Tuple[str, float]:
        """
        Adjust signal based on confirmation count.

        Returns:
            Tuple of (adjusted_signal, adjusted_confidence)
        """
        if signal == 'HOLD' or total == 0:
            return signal, confidence

        confirmation_rate = confirmations / total

        # Need at least min_confirming indicators
        if confirmations < self.min_confirming:
            # Convert to HOLD if not enough confirmation
            confidence_penalty = 0.5 + (confirmations / self.min_confirming) * 0.3
            return 'HOLD', confidence * confidence_penalty

        # Boost confidence based on confirmation rate
        if confirmation_rate >= 0.8:  # 80%+ agreement
            confidence *= 1.15
        elif confirmation_rate >= 0.6:  # 60-80% agreement
            confidence *= 1.05

        return signal, min(confidence, 100)


class TradingStrategy:
    """Enhanced trading strategy with adaptive thresholds, regime detection, and hysteresis"""

    def __init__(self):
        self.adaptive_thresholds = AdaptiveThresholds()
        self.regime_detector = RegimeDetector()
        self.hysteresis = SignalHysteresis(cooldown_periods=3)
        self.regime_filter = MarketRegimeFilter()
        self.multi_indicator = MultiIndicatorConfirmation(
            min_confirming=STRATEGY_CONFIG.get('min_confirming_indicators', 3)
        )
        self.min_confidence = STRATEGY_CONFIG['min_confidence']
        self.require_multi_indicator = STRATEGY_CONFIG.get('require_multi_indicator', True)
        self.signal_history = []

    def generate_signal(
        self,
        df: pd.DataFrame,
        lstm_prediction: str = None,
        lstm_confidence: float = None
    ) -> TradingSignal:
        """Generate trading signal (legacy interface)"""
        lstm_pred = None
        if lstm_prediction and lstm_confidence:
            lstm_pred = {
                'direction': lstm_prediction,
                'confidence': lstm_confidence * 100 if lstm_confidence <= 1 else lstm_confidence
            }
        return self.analyze(df, lstm_pred)

    def analyze(self, df: pd.DataFrame, lstm_prediction: Dict = None) -> TradingSignal:
        """Generate trading signal with comprehensive analysis"""
        if len(df) < 2:
            return TradingSignal(
                action='HOLD', confidence=0, price=0, reasons=['Insufficient data'],
                technical_score=0, lstm_score=0, regime='unknown'
            )

        current = df.iloc[-1]
        current_price = float(current['Close'])

        regime = self.regime_detector.detect(df)
        regime_adjustments = self.regime_detector.get_regime_adjustments(regime)
        thresholds = self.adaptive_thresholds.calculate(df)

        tech_score, tech_reasons = self._calculate_technical_score(df, current, thresholds, regime_adjustments)
        lstm_score, lstm_reasons = self._calculate_lstm_score(lstm_prediction)

        combined_score = tech_score * 0.6 + lstm_score * 0.4
        action, confidence = self._determine_action(combined_score, tech_score, lstm_score, regime_adjustments)
        action, confidence = self.hysteresis.filter_signal(action, confidence)

        # Apply market regime filter - block counter-trend signals
        if action != 'HOLD':
            action, confidence, regime_reasons = self.regime_filter.filter_signal(
                action, confidence, regime
            )
            tech_reasons.extend(regime_reasons)

        # Apply multi-indicator confirmation filter
        if action != 'HOLD' and self.require_multi_indicator:
            confirmations, total, confirm_reasons = self.multi_indicator.check_confirmation(df, action)
            action, confidence = self.multi_indicator.apply_confirmation_filter(
                action, confidence, confirmations, total
            )
            if confirm_reasons:
                tech_reasons.extend(confirm_reasons[:2])  # Add top 2 confirmation reasons

        if action != 'HOLD' and 'Volume_Ratio' in current:
            if current['Volume_Ratio'] < thresholds['volume_threshold']:
                confidence *= 0.85
                tech_reasons.append("Low volume - reduced confidence")

        all_reasons = tech_reasons + lstm_reasons
        stop_loss, take_profit = self._calculate_exit_levels(current_price, action, df, confidence)

        # Build indicators dictionary from current data
        indicators = {}
        indicator_columns = ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50', 'ATR', 'ADX', 'Stoch_K', 'Williams_R']
        for col in indicator_columns:
            if col in current and not pd.isna(current[col]):
                indicators[col] = float(current[col])
        # Map Stoch_K to Stochastic_K for compatibility
        if 'Stoch_K' in indicators:
            indicators['Stochastic_K'] = indicators['Stoch_K']

        signal = TradingSignal(
            action=action, confidence=round(confidence, 1), price=current_price,
            reasons=all_reasons[:5], technical_score=round(tech_score, 1),
            lstm_score=round(lstm_score, 1), regime=regime.value,
            stop_loss=stop_loss, take_profit=take_profit,
            indicators=indicators, market_regime=regime.value,
            trend_strength=abs(combined_score - 50) / 50
        )

        self.signal_history.append({'timestamp': df.index[-1] if hasattr(df.index, '__getitem__') else None, 'signal': signal})
        return signal

    def _calculate_technical_score(self, df: pd.DataFrame, current: pd.Series, thresholds: Dict, regime_adj: Dict) -> Tuple[float, List[str]]:
        """Calculate technical analysis score (0-100 scale, 50=neutral)"""
        score = 50
        reasons = []

        if 'RSI' in current and not pd.isna(current['RSI']):
            rsi = current['RSI']
            if rsi < thresholds['rsi_oversold']:
                score += 15
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > thresholds['rsi_overbought']:
                score -= 15
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif rsi < 50:
                score += 5
            else:
                score -= 5

        if 'MACD' in current and 'MACD_Signal' in current:
            macd = current['MACD']
            signal = current['MACD_Signal']
            if not pd.isna(macd) and not pd.isna(signal):
                if macd > signal:
                    score += 10
                    if len(df) > 1 and df['MACD'].iloc[-2] < df['MACD_Signal'].iloc[-2]:
                        score += 5
                        reasons.append("MACD bullish crossover")
                else:
                    score -= 10
                    if len(df) > 1 and df['MACD'].iloc[-2] > df['MACD_Signal'].iloc[-2]:
                        score -= 5
                        reasons.append("MACD bearish crossover")

        if 'SMA_20' in current and 'SMA_50' in current:
            sma_20, sma_50, close = current['SMA_20'], current['SMA_50'], current['Close']
            if not pd.isna(sma_20) and not pd.isna(sma_50):
                if close > sma_20 > sma_50:
                    score += 10
                    reasons.append("Price above rising SMAs")
                elif close < sma_20 < sma_50:
                    score -= 10
                    reasons.append("Price below falling SMAs")
                elif sma_20 > sma_50:
                    score += 5
                else:
                    score -= 5

        if 'Stoch_K' in current and 'Stoch_D' in current:
            stoch_k, stoch_d = current['Stoch_K'], current['Stoch_D']
            if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                if stoch_k < 20 and stoch_k > stoch_d:
                    score += 8
                    reasons.append("Stochastic bullish in oversold")
                elif stoch_k > 80 and stoch_k < stoch_d:
                    score -= 8
                    reasons.append("Stochastic bearish in overbought")

        if 'BB_Lower' in current and 'BB_Upper' in current:
            close, bb_lower, bb_upper = current['Close'], current['BB_Lower'], current['BB_Upper']
            if not pd.isna(bb_lower) and not pd.isna(bb_upper):
                if close < bb_lower:
                    score += 8
                    reasons.append("Price at lower Bollinger Band")
                elif close > bb_upper:
                    score -= 8
                    reasons.append("Price at upper Bollinger Band")

        if 'ADX' in current and not pd.isna(current['ADX']):
            if current['ADX'] > 25:
                reasons.append(f"Strong trend (ADX: {current['ADX']:.1f})")

        score += regime_adj['buy_bias'] * 20
        return max(0, min(100, score)), reasons

    def _calculate_lstm_score(self, lstm_prediction: Dict) -> Tuple[float, List[str]]:
        """Calculate LSTM prediction score"""
        if not lstm_prediction:
            return 50, []

        direction = lstm_prediction.get('direction', 'NEUTRAL')
        confidence = lstm_prediction.get('confidence', 0.5)
        if isinstance(confidence, (int, float)):
            confidence = confidence * 100 if confidence <= 1 else confidence

        reasons = []
        if direction == 'UP':
            score = 50 + (confidence * 0.5)
            reasons.append(f"LSTM predicts upward ({confidence:.0f}% conf)")
        elif direction == 'DOWN':
            score = 50 - (confidence * 0.5)
            reasons.append(f"LSTM predicts downward ({confidence:.0f}% conf)")
        else:
            score = 50
            reasons.append("LSTM predicts sideways")

        return min(100, max(0, score)), reasons

    def _determine_action(self, combined_score: float, tech_score: float, lstm_score: float, regime_adj: Dict) -> Tuple[str, float]:
        """Determine trading action from scores"""
        deviation = abs(combined_score - 50)
        confidence = min(50 + deviation, 95)
        confidence += regime_adj['confidence_boost'] * 100

        buy_threshold = 60 - regime_adj['buy_bias'] * 20
        sell_threshold = 40 + regime_adj['buy_bias'] * 20

        if combined_score > buy_threshold:
            action = 'BUY'
        elif combined_score < sell_threshold:
            action = 'SELL'
        else:
            action = 'HOLD'
            confidence *= 0.8

        if action != 'HOLD':
            tech_action = 'BUY' if tech_score > 55 else ('SELL' if tech_score < 45 else 'HOLD')
            lstm_action = 'BUY' if lstm_score > 55 else ('SELL' if lstm_score < 45 else 'HOLD')
            if tech_action != lstm_action and tech_action != 'HOLD' and lstm_action != 'HOLD':
                confidence *= 0.7

        return action, max(0, min(100, confidence))

    def _calculate_exit_levels(self, price: float, action: str, df: pd.DataFrame,
                                confidence: float = 75) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit levels with dynamic multipliers based on confidence.

        High confidence (85%+): Tighter SL, wider TP (SL=2.0×ATR, TP=4.0×ATR)
        Medium confidence (75%+): Moderate (SL=2.5×ATR, TP=3.5×ATR)
        Lower confidence: Wider SL, tighter TP (SL=3.0×ATR, TP=3.0×ATR)
        """
        if action == 'HOLD':
            return None, None

        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]) else price * 0.02

        # Dynamic multipliers based on confidence
        if confidence >= 85:
            # High confidence - tighter stop, wider target
            sl_mult = 2.0
            tp_mult = 4.0
        elif confidence >= 75:
            # Medium confidence - balanced
            sl_mult = 2.5
            tp_mult = 3.5
        else:
            # Lower confidence - wider stop, conservative target
            sl_mult = 3.0
            tp_mult = 3.0

        if action == 'BUY':
            stop_loss = price - (atr * sl_mult)
            take_profit = price + (atr * tp_mult)
        else:
            stop_loss = price + (atr * sl_mult)
            take_profit = price - (atr * tp_mult)

        return round(stop_loss, 2), round(take_profit, 2)

    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze the overall trend"""
        if len(df) < 20:
            return {'trend': 'INSUFFICIENT_DATA', 'strength': 0}

        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        price_5d = float(df['Close'].iloc[-5]) if len(df) >= 5 else current_price
        price_20d = float(df['Close'].iloc[-20]) if len(df) >= 20 else current_price

        change_5d = ((current_price - price_5d) / price_5d) * 100
        change_20d = ((current_price - price_20d) / price_20d) * 100

        if change_20d > 5 and change_5d > 0:
            trend, strength = 'STRONG_UPTREND', min(change_20d / 10, 1.0)
        elif change_20d > 0:
            trend, strength = 'UPTREND', min(change_20d / 10, 0.7)
        elif change_20d < -5 and change_5d < 0:
            trend, strength = 'STRONG_DOWNTREND', min(abs(change_20d) / 10, 1.0)
        elif change_20d < 0:
            trend, strength = 'DOWNTREND', min(abs(change_20d) / 10, 0.7)
        else:
            trend, strength = 'SIDEWAYS', 0.3

        return {'trend': trend, 'strength': strength, 'change_5d': round(change_5d, 2), 'change_20d': round(change_20d, 2), 'current_price': current_price}

    def get_support_resistance(self, df: pd.DataFrame, lookback: int = 30) -> Dict:
        """Calculate support and resistance levels"""
        if len(df) < lookback:
            lookback = len(df)

        recent_data = df.tail(lookback)
        high = float(recent_data['High'].max())
        low = float(recent_data['Low'].min())
        current = float(df['Close'].iloc[-1])

        pivot = (high + low + current) / 3
        r1, s1 = (2 * pivot) - low, (2 * pivot) - high
        r2, s2 = pivot + (high - low), pivot - (high - low)

        return {
            'current_price': current, 'resistance_1': round(r1, 2), 'resistance_2': round(r2, 2),
            'support_1': round(s1, 2), 'support_2': round(s2, 2), 'pivot': round(pivot, 2),
            'high': high, 'low': low
        }

    def signal_to_dict(self, signal: TradingSignal) -> Dict:
        """Convert TradingSignal to dictionary"""
        return {
            'action': signal.action, 'confidence': signal.confidence, 'price': signal.price,
            'reasons': signal.reasons, 'technical_score': signal.technical_score,
            'lstm_score': signal.lstm_score, 'regime': signal.regime,
            'stop_loss': signal.stop_loss, 'take_profit': signal.take_profit,
            'indicators': signal.indicators, 'market_regime': signal.market_regime,
            'trend_strength': signal.trend_strength
        }


def generate_trading_signal(df: pd.DataFrame, lstm_prediction: str = None, lstm_confidence: float = None) -> Dict:
    """Convenience function to generate trading signal"""
    strategy = TradingStrategy()
    signal = strategy.generate_signal(df, lstm_prediction, lstm_confidence)
    return strategy.signal_to_dict(signal)


if __name__ == "__main__":
    from data.data_loader import SaudiStockDataLoader
    from data.data_preprocessor import DataPreprocessor

    print("Testing Enhanced Trading Strategy...")
    print("=" * 60)

    loader = SaudiStockDataLoader()
    preprocessor = DataPreprocessor()

    raw_data = loader.fetch_stock_data("2222", period="6mo")
    clean_data = preprocessor.clean_data(raw_data)
    df = preprocessor.add_technical_indicators(clean_data)

    print(f"Analyzing {len(df)} days of data")

    lstm_pred = {'direction': 'UP', 'confidence': 72}

    strategy = TradingStrategy()
    signal = strategy.analyze(df, lstm_pred)

    print(f"\n--- Trading Signal ---")
    print(f"Action: {signal.action}")
    print(f"Confidence: {signal.confidence}%")
    print(f"Price: {signal.price:.2f} SAR")
    print(f"Regime: {signal.regime}")
    print(f"Technical Score: {signal.technical_score}")
    print(f"LSTM Score: {signal.lstm_score}")
    print(f"\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
    if signal.stop_loss:
        print(f"\nStop Loss: {signal.stop_loss:.2f} SAR")
        print(f"Take Profit: {signal.take_profit:.2f} SAR")
