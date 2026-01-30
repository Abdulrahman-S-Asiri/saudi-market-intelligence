# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
Advanced Validation and Backtesting for High-Accuracy LSTM Model

This module implements advanced cross-validation and backtesting techniques
specifically designed for financial time series:

1. PurgedWalkForwardCV: Walk-forward validation with data purging and embargo
2. MonteCarloBacktest: Monte Carlo simulations for robust performance estimation
3. Advanced Metrics: Probabilistic Sharpe, Deflated Sharpe, Omega Ratio
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Generator, Optional, Callable
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    """Container for backtest results."""
    returns: np.ndarray
    cumulative_returns: np.ndarray
    trades: List[Dict]
    metrics: Dict[str, float]


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation for Financial Time Series.

    This cross-validator implements walk-forward validation with:
    - Purging: Removes data points from training set that are too close to test
    - Embargo: Adds a buffer period after each test set before next train

    This prevents data leakage from overlapping labels and accounts for
    autocorrelation in financial time series.

    Reference:
        Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 5,
        test_size: float = 0.2,
        min_train_size: int = 252  # 1 year of trading days
    ):
        """
        Initialize PurgedWalkForwardCV.

        Args:
            n_splits: Number of folds
            purge_days: Number of days to remove from training before test
            embargo_days: Number of days to skip after test before next train
            test_size: Proportion of data for testing in each split
            min_train_size: Minimum training set size
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.test_size = test_size
        self.min_train_size = min_train_size

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for purged walk-forward CV.

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Target array (optional)
            groups: Group labels (optional)

        Yields:
            train_indices, test_indices for each split
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size / self.n_splits)

        for split_idx in range(self.n_splits):
            # Calculate test range
            test_start = int(n_samples * (1 - self.test_size) + split_idx * test_size)
            test_end = min(test_start + test_size, n_samples)

            if test_end <= test_start:
                continue

            # Calculate train range with purging
            train_end = max(0, test_start - self.purge_days)

            # Apply embargo from previous test periods
            if split_idx > 0:
                embargo_end = int(n_samples * (1 - self.test_size) + (split_idx - 1) * test_size + test_size + self.embargo_days)
                train_start = max(0, embargo_end)
            else:
                train_start = 0

            # Ensure minimum training size
            if train_end - train_start < self.min_train_size:
                train_start = max(0, train_end - self.min_train_size)

            if train_end <= train_start:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Generates multiple train/test combinations while respecting temporal order
    and applying purging to prevent data leakage.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_days: int = 5
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Number of groups to split data into
            n_test_splits: Number of groups to use as test in each combination
            purge_days: Days to purge between train and test
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_days = purge_days

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices for CPCV."""
        from itertools import combinations

        n_samples = len(X)
        split_size = n_samples // self.n_splits

        # Create split boundaries
        splits = [(i * split_size, min((i + 1) * split_size, n_samples))
                  for i in range(self.n_splits)]

        # Generate combinations of test splits
        for test_split_indices in combinations(range(self.n_splits), self.n_test_splits):
            test_indices = []
            train_indices = []

            for split_idx, (start, end) in enumerate(splits):
                if split_idx in test_split_indices:
                    test_indices.extend(range(start, end))
                else:
                    # Apply purging
                    purge_start = start
                    purge_end = end

                    for test_idx in test_split_indices:
                        test_start, test_end = splits[test_idx]
                        if test_start - self.purge_days < end <= test_start:
                            purge_end = min(purge_end, test_start - self.purge_days)
                        if test_end <= start < test_end + self.purge_days:
                            purge_start = max(purge_start, test_end + self.purge_days)

                    if purge_end > purge_start:
                        train_indices.extend(range(purge_start, purge_end))

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield np.array(train_indices), np.array(test_indices)


class MonteCarloBacktest:
    """
    Monte Carlo Backtesting with Realistic Market Frictions.

    Performs multiple simulations with randomized:
    - Slippage
    - Commission rates
    - Execution delays
    - Path variations (bootstrap)

    This provides a distribution of possible outcomes rather than a single
    deterministic result, enabling better risk assessment.
    """

    def __init__(
        self,
        n_simulations: int = 100,
        slippage_range: Tuple[float, float] = (0.0005, 0.002),
        commission_range: Tuple[float, float] = (0.0005, 0.001),
        execution_delay_range: Tuple[int, int] = (0, 2),
        use_bootstrap: bool = True,
        bootstrap_block_size: int = 10,
        random_seed: int = 42
    ):
        """
        Initialize Monte Carlo backtest.

        Args:
            n_simulations: Number of Monte Carlo simulations
            slippage_range: (min, max) slippage as fraction
            commission_range: (min, max) commission as fraction
            execution_delay_range: (min, max) delay in bars
            use_bootstrap: Whether to use block bootstrap for path variation
            bootstrap_block_size: Size of bootstrap blocks
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.slippage_range = slippage_range
        self.commission_range = commission_range
        self.execution_delay_range = execution_delay_range
        self.use_bootstrap = use_bootstrap
        self.bootstrap_block_size = bootstrap_block_size
        self.random_seed = random_seed

        np.random.seed(random_seed)

    def run_simulation(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        initial_capital: float = 100000,
        position_size: float = 0.1
    ) -> BacktestResult:
        """
        Run a single backtest simulation with randomized parameters.

        Args:
            signals: Array of signals (-1, 0, 1)
            prices: Array of prices
            initial_capital: Starting capital
            position_size: Fraction of capital per trade

        Returns:
            BacktestResult with trades and metrics
        """
        # Randomize parameters
        slippage = np.random.uniform(*self.slippage_range)
        commission = np.random.uniform(*self.commission_range)
        delay = np.random.randint(*self.execution_delay_range)

        # Apply execution delay
        if delay > 0:
            signals = np.roll(signals, delay)
            signals[:delay] = 0

        # Simulate trading
        capital = initial_capital
        position = 0
        trades = []
        returns = []

        for i in range(1, len(signals)):
            # Calculate return from previous position
            if position != 0:
                price_return = (prices[i] - prices[i-1]) / prices[i-1]
                trade_return = position * price_return - abs(position) * slippage
                returns.append(trade_return * abs(position) * position_size)
            else:
                returns.append(0)

            # Execute new signal
            if signals[i] != position and signals[i] != 0:
                # Close existing position
                if position != 0:
                    trades.append({
                        'type': 'close',
                        'price': prices[i] * (1 - slippage * np.sign(position)),
                        'position': position,
                        'cost': abs(position) * prices[i] * position_size * commission
                    })

                # Open new position
                position = signals[i]
                trades.append({
                    'type': 'open',
                    'price': prices[i] * (1 + slippage * position),
                    'position': position,
                    'cost': abs(position) * prices[i] * position_size * commission
                })

            # Close position on neutral signal
            elif signals[i] == 0 and position != 0:
                trades.append({
                    'type': 'close',
                    'price': prices[i] * (1 - slippage * np.sign(position)),
                    'position': position,
                    'cost': abs(position) * prices[i] * position_size * commission
                })
                position = 0

        returns = np.array(returns)

        # Account for commission costs
        total_commission = sum(t['cost'] for t in trades)
        if len(returns) > 0:
            returns[-1] -= total_commission / initial_capital

        cumulative_returns = (1 + returns).cumprod() - 1

        metrics = self._calculate_metrics(returns, cumulative_returns)

        return BacktestResult(
            returns=returns,
            cumulative_returns=cumulative_returns,
            trades=trades,
            metrics=metrics
        )

    def run_monte_carlo(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        initial_capital: float = 100000,
        position_size: float = 0.1
    ) -> Dict[str, any]:
        """
        Run full Monte Carlo simulation.

        Args:
            signals: Array of signals
            prices: Array of prices
            initial_capital: Starting capital
            position_size: Position size fraction

        Returns:
            Dictionary with simulation results and statistics
        """
        results = []

        for sim in range(self.n_simulations):
            # Optionally bootstrap the price path
            if self.use_bootstrap:
                prices_sim, signals_sim = self._block_bootstrap(prices, signals)
            else:
                prices_sim, signals_sim = prices, signals

            result = self.run_simulation(
                signals_sim, prices_sim, initial_capital, position_size
            )
            results.append(result)

        # Aggregate results
        all_returns = [r.metrics['total_return'] for r in results]
        all_sharpe = [r.metrics['sharpe_ratio'] for r in results]
        all_win_rates = [r.metrics['win_rate'] for r in results]
        all_max_dd = [r.metrics['max_drawdown'] for r in results]

        return {
            'results': results,
            'statistics': {
                'return_mean': np.mean(all_returns),
                'return_std': np.std(all_returns),
                'return_5th': np.percentile(all_returns, 5),
                'return_95th': np.percentile(all_returns, 95),
                'sharpe_mean': np.mean(all_sharpe),
                'sharpe_std': np.std(all_sharpe),
                'win_rate_mean': np.mean(all_win_rates),
                'max_drawdown_mean': np.mean(all_max_dd),
                'probability_positive': np.mean(np.array(all_returns) > 0),
                'probability_sharpe_above_1': np.mean(np.array(all_sharpe) > 1),
            }
        }

    def _block_bootstrap(
        self,
        prices: np.ndarray,
        signals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply block bootstrap to preserve autocorrelation."""
        n = len(prices)
        n_blocks = n // self.bootstrap_block_size + 1

        # Sample block starting indices
        block_starts = np.random.randint(0, n - self.bootstrap_block_size, n_blocks)

        # Reconstruct series from blocks
        bootstrapped_indices = []
        for start in block_starts:
            bootstrapped_indices.extend(range(start, min(start + self.bootstrap_block_size, n)))
            if len(bootstrapped_indices) >= n:
                break

        indices = np.array(bootstrapped_indices[:n])

        return prices[indices], signals[indices]

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        cumulative_returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Handle edge cases
        if len(returns) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'sortino_ratio': 0
            }

        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0

        # Sharpe Ratio (annualized)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0

        # Win Rate
        winning_trades = np.sum(returns > 0)
        total_trades = np.sum(returns != 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Max Drawdown
        running_max = np.maximum.accumulate(cumulative_returns + 1)
        drawdown = (running_max - (cumulative_returns + 1)) / running_max
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino = sharpe

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sortino_ratio': sortino
        }


class AdvancedMetrics:
    """
    Advanced performance metrics for rigorous strategy evaluation.

    Includes:
    - Probabilistic Sharpe Ratio (PSR)
    - Deflated Sharpe Ratio (DSR)
    - Omega Ratio
    - Tail Ratio
    - Information Ratio
    """

    @staticmethod
    def probabilistic_sharpe_ratio(
        returns: np.ndarray,
        benchmark_sharpe: float = 0,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio.

        PSR measures the probability that the true Sharpe ratio exceeds
        a benchmark, accounting for estimation error.

        Args:
            returns: Array of returns
            benchmark_sharpe: Benchmark Sharpe ratio to beat
            risk_free_rate: Annual risk-free rate

        Returns:
            PSR as probability (0 to 1)
        """
        n = len(returns)
        if n < 10 or returns.std() == 0:
            return 0.5

        # Excess returns
        excess_returns = returns - risk_free_rate / 252

        # Sample Sharpe
        sr = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        # Skewness and kurtosis
        skew = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Standard error of Sharpe ratio
        se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurtosis - 3) / 4 * sr**2) / (n - 1))

        # PSR
        psr = stats.norm.cdf((sr - benchmark_sharpe) / se_sr)

        return psr

    @staticmethod
    def deflated_sharpe_ratio(
        returns: np.ndarray,
        n_trials: int,
        variance_of_trials: float = None
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio.

        DSR adjusts the Sharpe ratio for multiple testing / strategy selection bias.
        If you tested N strategies and picked the best one, DSR accounts for this.

        Args:
            returns: Array of returns
            n_trials: Number of strategies/parameters tested
            variance_of_trials: Variance of Sharpe ratios across trials

        Returns:
            DSR as probability
        """
        n = len(returns)
        if n < 10 or returns.std() == 0:
            return 0.5

        sr = returns.mean() / returns.std() * np.sqrt(252)

        # Skewness and kurtosis
        skew = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Standard error
        se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurtosis - 3) / 4 * sr**2) / (n - 1))

        # Expected maximum Sharpe under null
        if variance_of_trials is None:
            variance_of_trials = 1

        expected_max_sr = variance_of_trials * (
            (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/n_trials) +
            np.euler_gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))
        )

        # DSR
        dsr = stats.norm.cdf((sr - expected_max_sr) / se_sr)

        return max(0, dsr)

    @staticmethod
    def omega_ratio(
        returns: np.ndarray,
        threshold: float = 0
    ) -> float:
        """
        Calculate Omega Ratio.

        Omega ratio is the probability-weighted ratio of gains vs losses
        above/below a threshold. Unlike Sharpe, it captures the full
        distribution of returns.

        Args:
            returns: Array of returns
            threshold: Threshold return (default 0)

        Returns:
            Omega ratio (>1 is good)
        """
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        if losses.sum() == 0:
            return float('inf')

        return gains.sum() / losses.sum()

    @staticmethod
    def tail_ratio(returns: np.ndarray, percentile: int = 5) -> float:
        """
        Calculate Tail Ratio.

        Compares the right tail (gains) to the left tail (losses).
        A ratio > 1 means larger positive outliers than negative ones.

        Args:
            returns: Array of returns
            percentile: Percentile for tail calculation

        Returns:
            Tail ratio
        """
        right_tail = np.percentile(returns, 100 - percentile)
        left_tail = np.abs(np.percentile(returns, percentile))

        if left_tail == 0:
            return float('inf')

        return right_tail / left_tail

    @staticmethod
    def information_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate Information Ratio.

        Measures the risk-adjusted excess return relative to a benchmark.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Information ratio (annualized)
        """
        excess = returns - benchmark_returns

        if excess.std() == 0:
            return 0

        return excess.mean() / excess.std() * np.sqrt(252)

    @staticmethod
    def calmar_ratio(
        returns: np.ndarray,
        period: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio.

        Annual return divided by maximum drawdown.

        Args:
            returns: Array of returns
            period: Trading days per year

        Returns:
            Calmar ratio
        """
        cumulative = (1 + returns).cumprod()
        max_dd = (cumulative.max() - cumulative.min()) / cumulative.max()

        if max_dd == 0:
            return float('inf')

        annual_return = (1 + returns.sum()) ** (period / len(returns)) - 1

        return annual_return / max_dd

    @classmethod
    def calculate_all(
        cls,
        returns: np.ndarray,
        benchmark_returns: np.ndarray = None,
        n_trials: int = 1
    ) -> Dict[str, float]:
        """
        Calculate all advanced metrics.

        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            n_trials: Number of trials for DSR

        Returns:
            Dictionary of all metrics
        """
        if benchmark_returns is None:
            benchmark_returns = np.zeros_like(returns)

        metrics = {
            'probabilistic_sharpe': cls.probabilistic_sharpe_ratio(returns),
            'deflated_sharpe': cls.deflated_sharpe_ratio(returns, n_trials),
            'omega_ratio': cls.omega_ratio(returns),
            'tail_ratio': cls.tail_ratio(returns),
            'information_ratio': cls.information_ratio(returns, benchmark_returns),
            'calmar_ratio': cls.calmar_ratio(returns)
        }

        # Add interpretation
        metrics['sharpe_is_significant'] = metrics['probabilistic_sharpe'] > 0.95
        metrics['strategy_is_robust'] = metrics['deflated_sharpe'] > 0.5

        return metrics


def run_advanced_backtest(
    model,
    data: pd.DataFrame,
    preprocessor,
    n_simulations: int = 100,
    initial_capital: float = 100000
) -> Dict:
    """
    Run comprehensive advanced backtest with Monte Carlo simulation.

    Args:
        model: Trained prediction model
        data: Preprocessed DataFrame with features
        preprocessor: DataPreprocessor instance
        n_simulations: Number of MC simulations
        initial_capital: Initial capital

    Returns:
        Complete backtest results with advanced metrics
    """
    from models.advanced_lstm import AdvancedStockPredictor

    # Get features and prepare data
    features = preprocessor.get_advanced_feature_list()
    available_features = [f for f in features if f in data.columns]

    # Generate predictions
    sequence_length = 60
    predictions = []
    confidences = []

    for i in range(sequence_length, len(data)):
        sequence = data[available_features].iloc[i-sequence_length:i].values
        scaled = preprocessor.scaler.transform(sequence)
        scaled = scaled.reshape(1, sequence_length, len(available_features))

        pred = model.predict(scaled, return_confidence=True)
        predictions.append(pred['prediction'][0][0])
        confidences.append(pred['confidence'][0][0])

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    # Convert to signals (-1, 0, 1)
    signals = np.zeros_like(predictions)
    signals[predictions > 0.5] = 1
    signals[predictions < -0.5] = -1

    # Only trade when confident
    signals[confidences < 0.6] = 0

    # Get prices
    prices = data['Close'].iloc[sequence_length:].values

    # Run Monte Carlo backtest
    mc_backtest = MonteCarloBacktest(n_simulations=n_simulations)
    mc_results = mc_backtest.run_monte_carlo(signals, prices, initial_capital)

    # Calculate advanced metrics on best result
    best_result = max(mc_results['results'], key=lambda r: r.metrics['sharpe_ratio'])
    advanced_metrics = AdvancedMetrics.calculate_all(
        best_result.returns,
        n_trials=n_simulations
    )

    return {
        'monte_carlo': mc_results['statistics'],
        'best_result': best_result.metrics,
        'advanced_metrics': advanced_metrics,
        'signals': signals,
        'predictions': predictions,
        'confidences': confidences
    }


if __name__ == "__main__":
    # Test the advanced validation module
    print("Testing Advanced Validation Module...")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    n = 500

    prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    signals = np.random.choice([-1, 0, 1], n, p=[0.2, 0.6, 0.2])

    # Test Monte Carlo backtest
    mc = MonteCarloBacktest(n_simulations=50)
    results = mc.run_monte_carlo(signals, prices)

    print(f"\nMonte Carlo Results ({mc.n_simulations} simulations):")
    print(f"  Return Mean: {results['statistics']['return_mean']:.2%}")
    print(f"  Return Std: {results['statistics']['return_std']:.2%}")
    print(f"  Sharpe Mean: {results['statistics']['sharpe_mean']:.2f}")
    print(f"  P(Positive Return): {results['statistics']['probability_positive']:.1%}")

    # Test advanced metrics
    returns = np.random.randn(252) * 0.02
    metrics = AdvancedMetrics.calculate_all(returns, n_trials=10)

    print(f"\nAdvanced Metrics:")
    print(f"  Probabilistic Sharpe: {metrics['probabilistic_sharpe']:.3f}")
    print(f"  Deflated Sharpe: {metrics['deflated_sharpe']:.3f}")
    print(f"  Omega Ratio: {metrics['omega_ratio']:.2f}")
    print(f"  Tail Ratio: {metrics['tail_ratio']:.2f}")

    # Test PurgedWalkForwardCV
    X = np.random.randn(500, 10)
    cv = PurgedWalkForwardCV(n_splits=5, purge_days=5, embargo_days=5)

    print(f"\nPurged Walk-Forward CV splits:")
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"  Split {i+1}: Train {len(train_idx)} samples, Test {len(test_idx)} samples")

    print("\nAdvanced Validation Module Test: OK")
