"""
Parameter optimizer for finding optimal trading strategy parameters.
Uses grid search to test different parameter combinations and find the best settings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product
import json
import os
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_engine import BacktestEngine, BacktestResult


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    best_params: Dict
    best_win_rate: float
    best_profit_factor: float
    best_sharpe: float
    all_results: List[Dict]
    optimization_date: str


class ParameterOptimizer:
    """
    Grid search optimizer for trading strategy parameters.
    Tests different combinations of min_confidence, hold_period, RSI thresholds, etc.
    """

    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_path = os.path.join(self.base_path, "backtest", "optimization_results")
        os.makedirs(self.results_path, exist_ok=True)

    def optimize(
        self,
        df: pd.DataFrame,
        param_grid: Dict[str, List] = None,
        target_metric: str = "win_rate",
        min_trades: int = 10
    ) -> OptimizationResult:
        """
        Run grid search optimization over parameter space.

        Args:
            df: DataFrame with price and indicator data
            param_grid: Dictionary of parameter names to lists of values to test
            target_metric: Metric to optimize ('win_rate', 'profit_factor', 'sharpe_ratio')
            min_trades: Minimum trades required for valid result

        Returns:
            OptimizationResult with best parameters and all results
        """
        if param_grid is None:
            param_grid = self.get_default_param_grid()

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"Testing {len(combinations)} parameter combinations...")

        all_results = []
        best_result = None
        best_metric_value = -float('inf')

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            # Run backtest with these parameters
            engine = BacktestEngine()
            try:
                result = engine.run(
                    df,
                    min_confidence=params.get('min_confidence', 75),
                    hold_period=params.get('hold_period', 10),
                    use_stop_loss=True
                )

                if result.total_trades >= min_trades:
                    result_dict = {
                        'params': params,
                        'win_rate': result.win_rate,
                        'profit_factor': result.profit_factor,
                        'sharpe_ratio': result.sharpe_ratio,
                        'sortino_ratio': result.sortino_ratio,
                        'total_trades': result.total_trades,
                        'total_return': result.total_return,
                        'max_drawdown': result.max_drawdown
                    }
                    all_results.append(result_dict)

                    # Check if this is the best result
                    metric_value = result_dict.get(target_metric, 0)
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result_dict

            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Tested {i + 1}/{len(combinations)} combinations...")

        if not best_result:
            raise ValueError("No valid results found. Try adjusting parameters or using more data.")

        optimization_result = OptimizationResult(
            best_params=best_result['params'],
            best_win_rate=best_result['win_rate'],
            best_profit_factor=best_result['profit_factor'],
            best_sharpe=best_result['sharpe_ratio'],
            all_results=all_results,
            optimization_date=datetime.now().isoformat()
        )

        return optimization_result

    def get_default_param_grid(self) -> Dict[str, List]:
        """Get default parameter grid for optimization"""
        return {
            'min_confidence': [70, 75, 80, 85],
            'hold_period': [5, 7, 10, 14],
        }

    def get_extended_param_grid(self) -> Dict[str, List]:
        """Get extended parameter grid for thorough optimization"""
        return {
            'min_confidence': [65, 70, 75, 80, 85, 90],
            'hold_period': [3, 5, 7, 10, 14, 21],
        }

    def optimize_for_stock(
        self,
        symbol: str,
        df: pd.DataFrame,
        param_grid: Dict[str, List] = None
    ) -> OptimizationResult:
        """
        Optimize parameters for a specific stock.

        Args:
            symbol: Stock symbol
            df: DataFrame with stock data
            param_grid: Parameter grid to search

        Returns:
            OptimizationResult for this stock
        """
        print(f"\nOptimizing parameters for {symbol}...")
        result = self.optimize(df, param_grid)

        # Save result
        self.save_result(symbol, result)

        return result

    def optimize_universal(
        self,
        stock_data: Dict[str, pd.DataFrame],
        param_grid: Dict[str, List] = None
    ) -> OptimizationResult:
        """
        Find universal parameters that work across multiple stocks.

        Args:
            stock_data: Dictionary of symbol -> DataFrame
            param_grid: Parameter grid to search

        Returns:
            OptimizationResult with best universal parameters
        """
        if param_grid is None:
            param_grid = self.get_default_param_grid()

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"Finding universal parameters across {len(stock_data)} stocks...")
        print(f"Testing {len(combinations)} parameter combinations...")

        universal_results = []

        for combo in combinations:
            params = dict(zip(param_names, combo))
            stock_results = []

            for symbol, df in stock_data.items():
                engine = BacktestEngine()
                try:
                    result = engine.run(
                        df,
                        min_confidence=params.get('min_confidence', 75),
                        hold_period=params.get('hold_period', 10),
                        use_stop_loss=True
                    )

                    if result.total_trades >= 5:
                        stock_results.append({
                            'symbol': symbol,
                            'win_rate': result.win_rate,
                            'profit_factor': result.profit_factor,
                            'total_trades': result.total_trades
                        })
                except Exception:
                    continue

            if stock_results:
                avg_win_rate = np.mean([r['win_rate'] for r in stock_results])
                avg_profit_factor = np.mean([r['profit_factor'] for r in stock_results])
                min_win_rate = min(r['win_rate'] for r in stock_results)

                universal_results.append({
                    'params': params,
                    'avg_win_rate': avg_win_rate,
                    'avg_profit_factor': avg_profit_factor,
                    'min_win_rate': min_win_rate,
                    'stocks_tested': len(stock_results),
                    'stock_results': stock_results
                })

        # Find best universal parameters (optimize for consistency)
        # Use a combined metric: avg_win_rate * min_win_rate to ensure consistency
        best = max(universal_results, key=lambda x: x['avg_win_rate'] * (x['min_win_rate'] / 100))

        return OptimizationResult(
            best_params=best['params'],
            best_win_rate=best['avg_win_rate'],
            best_profit_factor=best['avg_profit_factor'],
            best_sharpe=0,  # Not calculated for universal
            all_results=universal_results,
            optimization_date=datetime.now().isoformat()
        )

    def save_result(self, symbol: str, result: OptimizationResult) -> str:
        """Save optimization result to file"""
        filename = f"optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_path, filename)

        data = {
            'symbol': symbol,
            'best_params': result.best_params,
            'best_win_rate': result.best_win_rate,
            'best_profit_factor': result.best_profit_factor,
            'best_sharpe': result.best_sharpe,
            'optimization_date': result.optimization_date,
            'top_results': sorted(result.all_results, key=lambda x: x['win_rate'], reverse=True)[:10]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {filepath}")
        return filepath

    def load_best_params(self, symbol: str) -> Optional[Dict]:
        """Load best parameters for a symbol from saved results"""
        files = [f for f in os.listdir(self.results_path) if f.startswith(f"optimization_{symbol}_")]

        if not files:
            return None

        # Get most recent
        latest = sorted(files)[-1]
        filepath = os.path.join(self.results_path, latest)

        with open(filepath, 'r') as f:
            data = json.load(f)

        return data.get('best_params')

    def print_optimization_summary(self, result: OptimizationResult):
        """Print a summary of optimization results"""
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"\nBest Parameters:")
        for key, value in result.best_params.items():
            print(f"  {key}: {value}")
        print(f"\nBest Metrics:")
        print(f"  Win Rate: {result.best_win_rate:.1f}%")
        print(f"  Profit Factor: {result.best_profit_factor:.2f}")
        print(f"  Sharpe Ratio: {result.best_sharpe:.2f}")
        print(f"\nTotal combinations tested: {len(result.all_results)}")

        # Top 5 results
        print("\nTop 5 Parameter Combinations:")
        top5 = sorted(result.all_results, key=lambda x: x['win_rate'], reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r['params'].items())
            print(f"  {i}. Win Rate: {r['win_rate']:.1f}% | {params_str}")


if __name__ == "__main__":
    from data.data_loader import SaudiStockDataLoader
    from data.data_preprocessor import DataPreprocessor

    print("Testing Parameter Optimizer...")
    print("=" * 60)

    loader = SaudiStockDataLoader()
    preprocessor = DataPreprocessor()

    # Load data
    raw_data = loader.fetch_stock_data("2222", period="1y")
    clean_data = preprocessor.clean_data(raw_data)
    df = preprocessor.add_technical_indicators(clean_data)

    print(f"Loaded {len(df)} days of data for optimization")

    # Run optimization
    optimizer = ParameterOptimizer()
    result = optimizer.optimize(df, target_metric='win_rate')

    # Print summary
    optimizer.print_optimization_summary(result)
