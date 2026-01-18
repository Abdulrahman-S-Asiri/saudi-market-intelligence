"""
Win Rate Tracker for tracking progress toward 80% win rate goal.
Records experiment results and generates progress reports.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ExperimentRecord:
    """Record of a single backtest experiment"""
    experiment_id: str
    timestamp: str
    symbol: str
    win_rate: float
    total_trades: int
    profit_factor: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    parameters: Dict
    notes: str = ""


class WinRateTracker:
    """
    Track win rates across experiments and show progress toward 80% goal.
    """

    TARGET_WIN_RATE = 80.0

    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tracker_path = os.path.join(self.base_path, "backtest", "win_rate_history")
        self.history_file = os.path.join(self.tracker_path, "experiment_history.json")
        os.makedirs(self.tracker_path, exist_ok=True)

        self.experiments: List[ExperimentRecord] = []
        self._load_history()

    def _load_history(self):
        """Load experiment history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                self.experiments = [
                    ExperimentRecord(**exp) for exp in data.get('experiments', [])
                ]
            except Exception as e:
                print(f"Warning: Could not load history: {e}")
                self.experiments = []

    def _save_history(self):
        """Save experiment history to file"""
        data = {
            'last_updated': datetime.now().isoformat(),
            'total_experiments': len(self.experiments),
            'target_win_rate': self.TARGET_WIN_RATE,
            'experiments': [asdict(exp) for exp in self.experiments]
        }
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record_experiment(
        self,
        symbol: str,
        win_rate: float,
        total_trades: int,
        profit_factor: float,
        sharpe_ratio: float,
        total_return: float,
        max_drawdown: float,
        parameters: Dict,
        notes: str = ""
    ) -> ExperimentRecord:
        """
        Record a new backtest experiment result.

        Args:
            symbol: Stock symbol tested
            win_rate: Win rate percentage
            total_trades: Total number of trades
            profit_factor: Profit factor
            sharpe_ratio: Sharpe ratio
            total_return: Total return percentage
            max_drawdown: Maximum drawdown percentage
            parameters: Dictionary of parameters used
            notes: Optional notes about the experiment

        Returns:
            ExperimentRecord that was created
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.experiments)}"

        record = ExperimentRecord(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            win_rate=round(win_rate, 2),
            total_trades=total_trades,
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            total_return=round(total_return, 2),
            max_drawdown=round(max_drawdown, 2),
            parameters=parameters,
            notes=notes
        )

        self.experiments.append(record)
        self._save_history()

        # Print progress update
        self._print_progress_update(record)

        return record

    def _print_progress_update(self, record: ExperimentRecord):
        """Print progress update after recording an experiment"""
        gap = self.TARGET_WIN_RATE - record.win_rate
        status = "TARGET ACHIEVED!" if gap <= 0 else f"{gap:.1f}% to go"

        print(f"\n{'=' * 50}")
        print(f"EXPERIMENT RECORDED: {record.experiment_id}")
        print(f"{'=' * 50}")
        print(f"Symbol: {record.symbol}")
        print(f"Win Rate: {record.win_rate:.1f}% ({status})")
        print(f"Trades: {record.total_trades}")
        print(f"Profit Factor: {record.profit_factor:.2f}")
        print(f"Total Return: {record.total_return:.2f}%")
        print(f"{'=' * 50}")

    def get_best_experiment(self, symbol: str = None) -> Optional[ExperimentRecord]:
        """Get the best experiment by win rate"""
        experiments = self.experiments
        if symbol:
            experiments = [e for e in experiments if e.symbol == symbol]

        if not experiments:
            return None

        return max(experiments, key=lambda x: x.win_rate)

    def get_progress_summary(self) -> Dict:
        """Get summary of progress toward 80% goal"""
        if not self.experiments:
            return {
                'total_experiments': 0,
                'target_win_rate': self.TARGET_WIN_RATE,
                'best_win_rate': 0,
                'gap_to_target': self.TARGET_WIN_RATE,
                'target_achieved': False
            }

        best = max(self.experiments, key=lambda x: x.win_rate)
        avg_win_rate = np.mean([e.win_rate for e in self.experiments])

        # Get recent trend (last 5 experiments)
        recent = self.experiments[-5:]
        recent_avg = np.mean([e.win_rate for e in recent]) if recent else 0

        # Check improvement trend
        if len(self.experiments) >= 10:
            first_half = self.experiments[:len(self.experiments)//2]
            second_half = self.experiments[len(self.experiments)//2:]
            first_avg = np.mean([e.win_rate for e in first_half])
            second_avg = np.mean([e.win_rate for e in second_half])
            trend = "Improving" if second_avg > first_avg else "Declining"
        else:
            trend = "Insufficient data"

        return {
            'total_experiments': len(self.experiments),
            'target_win_rate': self.TARGET_WIN_RATE,
            'best_win_rate': best.win_rate,
            'best_experiment_id': best.experiment_id,
            'best_symbol': best.symbol,
            'avg_win_rate': round(avg_win_rate, 2),
            'recent_avg_win_rate': round(recent_avg, 2),
            'gap_to_target': round(self.TARGET_WIN_RATE - best.win_rate, 2),
            'target_achieved': best.win_rate >= self.TARGET_WIN_RATE,
            'trend': trend
        }

    def get_symbol_summary(self, symbol: str) -> Dict:
        """Get summary for a specific symbol"""
        symbol_exps = [e for e in self.experiments if e.symbol == symbol]

        if not symbol_exps:
            return {'symbol': symbol, 'experiments': 0}

        best = max(symbol_exps, key=lambda x: x.win_rate)

        return {
            'symbol': symbol,
            'experiments': len(symbol_exps),
            'best_win_rate': best.win_rate,
            'avg_win_rate': round(np.mean([e.win_rate for e in symbol_exps]), 2),
            'best_parameters': best.parameters,
            'target_achieved': best.win_rate >= self.TARGET_WIN_RATE
        }

    def print_full_report(self):
        """Print comprehensive progress report"""
        summary = self.get_progress_summary()

        print("\n" + "=" * 60)
        print("WIN RATE TRACKER - PROGRESS REPORT")
        print("=" * 60)

        print(f"\nTarget: {self.TARGET_WIN_RATE}% Win Rate")
        print(f"Experiments Run: {summary['total_experiments']}")

        if summary['total_experiments'] > 0:
            print(f"\nBest Win Rate: {summary['best_win_rate']:.1f}%")
            print(f"  Experiment: {summary['best_experiment_id']}")
            print(f"  Symbol: {summary['best_symbol']}")

            print(f"\nAverage Win Rate: {summary['avg_win_rate']:.1f}%")
            print(f"Recent Average (last 5): {summary['recent_avg_win_rate']:.1f}%")

            if summary['target_achieved']:
                print(f"\n*** TARGET ACHIEVED! ***")
            else:
                print(f"\nGap to Target: {summary['gap_to_target']:.1f}%")

            print(f"Trend: {summary['trend']}")

            # Top 5 experiments
            print(f"\nTop 5 Experiments:")
            top5 = sorted(self.experiments, key=lambda x: x.win_rate, reverse=True)[:5]
            for i, exp in enumerate(top5, 1):
                status = "*" if exp.win_rate >= self.TARGET_WIN_RATE else " "
                print(f"  {status}{i}. {exp.win_rate:.1f}% - {exp.symbol} - {exp.total_trades} trades")

            # By symbol
            print(f"\nBy Symbol:")
            symbols = set(e.symbol for e in self.experiments)
            for symbol in sorted(symbols):
                sym_summary = self.get_symbol_summary(symbol)
                status = "*" if sym_summary['target_achieved'] else " "
                print(f"  {status}{symbol}: Best {sym_summary['best_win_rate']:.1f}% "
                      f"(Avg: {sym_summary['avg_win_rate']:.1f}%, {sym_summary['experiments']} exp)")

        print("\n" + "=" * 60)

    def get_recommendations(self) -> List[str]:
        """Get recommendations for improving win rate"""
        recommendations = []

        if not self.experiments:
            recommendations.append("Run initial backtests to establish baseline")
            return recommendations

        summary = self.get_progress_summary()

        if summary['target_achieved']:
            recommendations.append("Target achieved! Consider testing on more stocks for robustness")
        else:
            gap = summary['gap_to_target']

            if gap > 20:
                recommendations.append("Large gap - consider fundamental strategy changes")
                recommendations.append("Try increasing min_confidence to 85+")
                recommendations.append("Add more indicator confirmations")
            elif gap > 10:
                recommendations.append("Moderate gap - fine-tune parameters")
                recommendations.append("Use parameter optimizer to find better settings")
                recommendations.append("Consider longer hold periods")
            else:
                recommendations.append("Close to target - small adjustments needed")
                recommendations.append("Focus on filtering out lowest confidence signals")

            # Check if specific signal type is weak
            recent = self.experiments[-10:]
            if recent:
                avg = np.mean([e.win_rate for e in recent])
                if avg < 60:
                    recommendations.append("Recent performance declining - review market conditions")

        return recommendations

    def clear_history(self):
        """Clear all experiment history"""
        self.experiments = []
        self._save_history()
        print("Experiment history cleared")


if __name__ == "__main__":
    print("Testing Win Rate Tracker...")
    print("=" * 60)

    tracker = WinRateTracker()

    # Record some test experiments
    tracker.record_experiment(
        symbol="2222",
        win_rate=65.5,
        total_trades=45,
        profit_factor=1.8,
        sharpe_ratio=1.2,
        total_return=15.5,
        max_drawdown=8.2,
        parameters={'min_confidence': 75, 'hold_period': 10},
        notes="Test run with new parameters"
    )

    # Print full report
    tracker.print_full_report()

    # Get recommendations
    print("\nRecommendations:")
    for rec in tracker.get_recommendations():
        print(f"  - {rec}")
