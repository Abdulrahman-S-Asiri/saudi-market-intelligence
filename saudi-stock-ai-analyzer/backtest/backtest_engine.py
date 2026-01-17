"""
Backtesting engine for strategy evaluation
Historical signal simulation with performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.trading_strategy import TradingStrategy
from backtest.performance_metrics import PerformanceMetrics


@dataclass
class Trade:
    """Individual trade record"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    signal_type: str  # BUY or SELL
    confidence: float
    return_pct: float
    profit: float
    holding_days: int
    correct: bool


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    avg_return_per_trade: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_holding_days: float
    trades: List[Trade]


class BacktestEngine:
    """
    Backtesting engine for strategy evaluation
    Simulates historical signals and calculates performance
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        position_size: float = 0.1,
        commission: float = 0.001,
        slippage: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage
        self.strategy = TradingStrategy()
        self.metrics = PerformanceMetrics()

    def run(
        self,
        df: pd.DataFrame,
        min_confidence: float = 60,
        hold_period: int = 5,
        use_stop_loss: bool = True
    ) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            df: DataFrame with price and indicator data
            min_confidence: Minimum confidence to take a trade
            hold_period: Default holding period in days
            use_stop_loss: Whether to use stop loss/take profit

        Returns:
            BacktestResult with comprehensive performance metrics
        """
        trades = []
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        position = None
        position_entry_idx = None

        # Walk through data
        for i in range(60, len(df) - hold_period):
            current_data = df.iloc[:i+1]
            current_price = float(df.iloc[i]['Close'])

            # Check if we have an open position
            if position is not None:
                days_held = i - position_entry_idx
                current_pnl = self._calculate_pnl(
                    position['entry_price'],
                    current_price,
                    position['type']
                )

                # Check exit conditions
                should_exit = False
                exit_reason = None

                # Hold period exit
                if days_held >= hold_period:
                    should_exit = True
                    exit_reason = 'hold_period'

                # Stop loss/take profit
                if use_stop_loss and position.get('stop_loss') and position.get('take_profit'):
                    if position['type'] == 'BUY':
                        if current_price <= position['stop_loss']:
                            should_exit = True
                            exit_reason = 'stop_loss'
                        elif current_price >= position['take_profit']:
                            should_exit = True
                            exit_reason = 'take_profit'
                    else:  # SELL
                        if current_price >= position['stop_loss']:
                            should_exit = True
                            exit_reason = 'stop_loss'
                        elif current_price <= position['take_profit']:
                            should_exit = True
                            exit_reason = 'take_profit'

                if should_exit:
                    # Close position
                    exit_price = current_price * (1 - self.slippage if position['type'] == 'SELL' else 1 + self.slippage)
                    return_pct = self._calculate_pnl(position['entry_price'], exit_price, position['type'])
                    profit = position['size'] * return_pct / 100

                    trade = Trade(
                        entry_date=str(df.index[position_entry_idx]),
                        exit_date=str(df.index[i]),
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        signal_type=position['type'],
                        confidence=position['confidence'],
                        return_pct=return_pct,
                        profit=profit,
                        holding_days=days_held,
                        correct=return_pct > 0
                    )
                    trades.append(trade)

                    current_capital += profit - (position['size'] * self.commission * 2)
                    position = None
                    position_entry_idx = None

            else:
                # Generate signal for new position
                signal = self.strategy.analyze(current_data)

                if signal.confidence >= min_confidence and signal.action != 'HOLD':
                    # Calculate position size
                    position_value = current_capital * self.position_size
                    entry_price = current_price * (1 + self.slippage if signal.action == 'BUY' else 1 - self.slippage)

                    position = {
                        'type': signal.action,
                        'entry_price': entry_price,
                        'size': position_value,
                        'confidence': signal.confidence,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit
                    }
                    position_entry_idx = i

            equity_curve.append(current_capital)

        # Calculate results
        return self._calculate_results(trades, equity_curve)

    def _calculate_pnl(self, entry: float, exit: float, trade_type: str) -> float:
        """Calculate percentage P&L"""
        if trade_type == 'BUY':
            return ((exit - entry) / entry) * 100
        else:  # SELL
            return ((entry - exit) / entry) * 100

    def _calculate_results(self, trades: List[Trade], equity_curve: List[float]) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        if not trades:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, profit_factor=0, total_return=0,
                avg_return_per_trade=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, avg_holding_days=0, trades=[]
            )

        # Basic stats
        winning_trades = [t for t in trades if t.correct]
        losing_trades = [t for t in trades if not t.correct]

        total_gains = sum(t.profit for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.profit for t in losing_trades)) if losing_trades else 0.0001

        # Returns
        returns = [t.return_pct for t in trades]
        total_return = ((equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100

        # Max drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Sharpe and Sortino ratios
        returns_array = np.array(returns)
        if len(returns_array) > 1 and np.std(returns_array) > 0:
            sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252 / 5)  # Annualized
            downside_returns = returns_array[returns_array < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0001
            sortino = np.mean(returns_array) / downside_std * np.sqrt(252 / 5)
        else:
            sharpe = 0
            sortino = 0

        return BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=(len(winning_trades) / len(trades)) * 100 if trades else 0,
            profit_factor=total_gains / total_losses if total_losses > 0 else 0,
            total_return=round(total_return, 2),
            avg_return_per_trade=round(np.mean(returns), 2) if returns else 0,
            max_drawdown=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            avg_holding_days=round(np.mean([t.holding_days for t in trades]), 1) if trades else 0,
            trades=trades
        )

    def run_signal_backtest(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5
    ) -> Dict:
        """
        Evaluate signal accuracy without position simulation

        Args:
            df: DataFrame with data
            forward_periods: Periods to look forward for outcome

        Returns:
            Signal accuracy metrics
        """
        signals = []
        correct_signals = 0
        total_signals = 0

        for i in range(60, len(df) - forward_periods):
            current_data = df.iloc[:i+1]
            signal = self.strategy.analyze(current_data)

            if signal.action == 'HOLD':
                continue

            current_price = float(df.iloc[i]['Close'])
            future_price = float(df.iloc[i + forward_periods]['Close'])
            actual_return = ((future_price - current_price) / current_price) * 100

            correct = (signal.action == 'BUY' and actual_return > 0) or \
                     (signal.action == 'SELL' and actual_return < 0)

            signals.append({
                'date': str(df.index[i]),
                'signal': signal.action,
                'confidence': signal.confidence,
                'price': current_price,
                'future_price': future_price,
                'actual_return': round(actual_return, 2),
                'correct': correct
            })

            if correct:
                correct_signals += 1
            total_signals += 1

        accuracy = (correct_signals / total_signals * 100) if total_signals > 0 else 0

        # Accuracy by confidence level
        high_conf_signals = [s for s in signals if s['confidence'] >= 70]
        high_conf_correct = sum(1 for s in high_conf_signals if s['correct'])
        high_conf_accuracy = (high_conf_correct / len(high_conf_signals) * 100) if high_conf_signals else 0

        # Accuracy by signal type
        buy_signals = [s for s in signals if s['signal'] == 'BUY']
        sell_signals = [s for s in signals if s['signal'] == 'SELL']
        buy_accuracy = (sum(1 for s in buy_signals if s['correct']) / len(buy_signals) * 100) if buy_signals else 0
        sell_accuracy = (sum(1 for s in sell_signals if s['correct']) / len(sell_signals) * 100) if sell_signals else 0

        return {
            'total_signals': total_signals,
            'correct_signals': correct_signals,
            'accuracy': round(accuracy, 1),
            'high_confidence_accuracy': round(high_conf_accuracy, 1),
            'buy_accuracy': round(buy_accuracy, 1),
            'sell_accuracy': round(sell_accuracy, 1),
            'avg_return_correct': round(np.mean([s['actual_return'] for s in signals if s['correct']]), 2) if signals else 0,
            'avg_return_incorrect': round(np.mean([s['actual_return'] for s in signals if not s['correct']]), 2) if signals else 0,
            'signals': signals[-20:]  # Last 20 signals
        }

    def result_to_dict(self, result: BacktestResult) -> Dict:
        """Convert BacktestResult to dictionary"""
        return {
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_return': result.total_return,
            'avg_return_per_trade': result.avg_return_per_trade,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'avg_holding_days': result.avg_holding_days,
            'trades': [{
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'signal_type': t.signal_type,
                'return_pct': t.return_pct,
                'correct': t.correct
            } for t in result.trades[-10:]]  # Last 10 trades
        }


if __name__ == "__main__":
    from data.data_loader import SaudiStockDataLoader
    from data.data_preprocessor import DataPreprocessor

    print("Testing Backtest Engine...")
    print("=" * 60)

    loader = SaudiStockDataLoader()
    preprocessor = DataPreprocessor()

    raw_data = loader.fetch_stock_data("2222", period="1y")
    clean_data = preprocessor.clean_data(raw_data)
    df = preprocessor.add_technical_indicators(clean_data)

    print(f"Backtesting on {len(df)} days of data")

    engine = BacktestEngine()

    # Run full backtest
    result = engine.run(df, min_confidence=60, hold_period=5)

    print(f"\n--- Backtest Results ---")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total Return: {result.total_return:.2f}%")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.2f}")

    # Run signal accuracy test
    accuracy = engine.run_signal_backtest(df)
    print(f"\n--- Signal Accuracy ---")
    print(f"Total Signals: {accuracy['total_signals']}")
    print(f"Accuracy: {accuracy['accuracy']}%")
    print(f"High Confidence Accuracy: {accuracy['high_confidence_accuracy']}%")
    print(f"Buy Accuracy: {accuracy['buy_accuracy']}%")
    print(f"Sell Accuracy: {accuracy['sell_accuracy']}%")
