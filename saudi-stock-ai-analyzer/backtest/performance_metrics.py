"""
Performance metrics calculation and reporting
Calculates various trading and model performance metrics
Includes advanced risk metrics: Sortino, Calmar, VaR, Expected Shortfall
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PerformanceReport:
    """Performance metrics report"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_daily_return: float
    best_day: float
    worst_day: float
    trading_days: int


@dataclass
class RiskReport:
    """Comprehensive risk metrics report"""
    # Basic risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # Days in drawdown

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Value at Risk
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% Expected Shortfall
    cvar_99: float  # 99% Expected Shortfall

    # Distribution metrics
    skewness: float
    kurtosis: float

    # Downside metrics
    downside_deviation: float
    ulcer_index: float
    pain_index: float


class PerformanceMetrics:
    """
    Calculate performance metrics for stock analysis
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize with risk-free rate

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate daily returns from price data"""
        if 'Daily_Return' in df.columns:
            return df['Daily_Return'].dropna()
        return df['Close'].pct_change().dropna()

    def calculate_total_return(self, df: pd.DataFrame) -> float:
        """Calculate total return over the period"""
        if len(df) < 1:
            return 0.0
        start_price = float(df['Close'].iloc[0])
        end_price = float(df['Close'].iloc[-1])
        if start_price == 0:
            return 0.0
        return ((end_price - start_price) / start_price) * 100

    def calculate_annualized_return(self, df: pd.DataFrame) -> float:
        """Calculate annualized return"""
        total_return = self.calculate_total_return(df) / 100
        trading_days = len(df)
        years = trading_days / 252  # Approximate trading days per year

        if years <= 0:
            return 0

        annualized = ((1 + total_return) ** (1 / years) - 1) * 100
        return annualized

    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        returns = self.calculate_returns(df)
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252) * 100
        return annualized_vol

    def calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        annualized_return = self.calculate_annualized_return(df) / 100
        volatility = self.calculate_volatility(df) / 100

        if volatility == 0:
            return 0

        sharpe = (annualized_return - self.risk_free_rate) / volatility
        return sharpe

    def calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        prices = df['Close']
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        return abs(max_dd)

    def calculate_win_rate(self, df: pd.DataFrame) -> float:
        """Calculate win rate (percentage of positive days)"""
        returns = self.calculate_returns(df)
        positive_days = (returns > 0).sum()
        total_days = len(returns)

        if total_days == 0:
            return 0

        return (positive_days / total_days) * 100

    def calculate_model_accuracy(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict:
        """
        Calculate model prediction accuracy

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            Dictionary with accuracy metrics
        """
        # Direction accuracy
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actuals))
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100

        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - actuals))

        # Root Mean Square Error
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        return {
            'direction_accuracy': direction_accuracy,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    def generate_report(self, df: pd.DataFrame) -> PerformanceReport:
        """
        Generate comprehensive performance report

        Args:
            df: DataFrame with price data

        Returns:
            PerformanceReport object
        """
        if len(df) < 2:
            return PerformanceReport(
                total_return=0, annualized_return=0, volatility=0,
                sharpe_ratio=0, max_drawdown=0, win_rate=0,
                avg_daily_return=0, best_day=0, worst_day=0, trading_days=len(df)
            )
        returns = self.calculate_returns(df)

        if len(returns) < 1:
            return PerformanceReport(
                total_return=0, annualized_return=0, volatility=0,
                sharpe_ratio=0, max_drawdown=0, win_rate=0,
                avg_daily_return=0, best_day=0, worst_day=0, trading_days=len(df)
            )

        return PerformanceReport(
            total_return=self.calculate_total_return(df),
            annualized_return=self.calculate_annualized_return(df),
            volatility=self.calculate_volatility(df),
            sharpe_ratio=self.calculate_sharpe_ratio(df),
            max_drawdown=self.calculate_max_drawdown(df),
            win_rate=self.calculate_win_rate(df),
            avg_daily_return=float(returns.mean() * 100),
            best_day=float(returns.max() * 100),
            worst_day=float(returns.min() * 100),
            trading_days=len(df)
        )

    def report_to_dict(self, report: PerformanceReport) -> Dict:
        """Convert PerformanceReport to dictionary"""
        return {
            'total_return': round(report.total_return, 2),
            'annualized_return': round(report.annualized_return, 2),
            'volatility': round(report.volatility, 2),
            'sharpe_ratio': round(report.sharpe_ratio, 2),
            'max_drawdown': round(report.max_drawdown, 2),
            'win_rate': round(report.win_rate, 2),
            'avg_daily_return': round(report.avg_daily_return, 4),
            'best_day': round(report.best_day, 2),
            'worst_day': round(report.worst_day, 2),
            'trading_days': report.trading_days
        }


class RiskMetrics:
    """
    Advanced risk metrics calculation
    Includes VaR, Expected Shortfall, Sortino, Calmar ratios
    """

    def __init__(self, risk_free_rate: float = 0.02, target_return: float = 0.0):
        """
        Initialize risk metrics calculator

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            target_return: Target return for downside calculations (default 0%)
        """
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return

    def calculate_returns(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate daily returns"""
        if 'Daily_Return' in df.columns:
            returns = df['Daily_Return'].dropna().values
        else:
            returns = df['Close'].pct_change().dropna().values
        return returns

    def calculate_sortino_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate Sortino ratio
        Uses downside deviation instead of standard deviation
        Better for measuring risk-adjusted returns with asymmetric distributions
        """
        if len(df) < 2:
            return 0.0
        returns = self.calculate_returns(df)

        # Annualized return
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        years = len(df) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Downside deviation (only negative returns below target)
        downside_returns = returns[returns < self.target_return]
        if len(downside_returns) < 2:
            return 0

        downside_std = np.std(downside_returns)
        annualized_downside = downside_std * np.sqrt(252)

        if annualized_downside == 0:
            return 0

        sortino = (annualized_return - self.risk_free_rate) / annualized_downside
        return sortino

    def calculate_calmar_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate Calmar ratio
        Annualized return divided by maximum drawdown
        Good for assessing performance relative to drawdown risk
        """
        if len(df) < 2:
            return 0.0
        # Annualized return
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        years = len(df) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Maximum drawdown
        prices = df['Close'].values
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        max_drawdown = np.max(drawdown)

        if max_drawdown == 0:
            return 0

        calmar = annualized_return / max_drawdown
        return calmar

    def calculate_var(
        self,
        df: pd.DataFrame,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        Maximum expected loss at given confidence level

        Args:
            df: DataFrame with price data
            confidence: Confidence level (0.95 = 95%)
            method: 'historical', 'parametric', or 'cornish_fisher'

        Returns:
            VaR as percentage (negative value indicating loss)
        """
        returns = self.calculate_returns(df)

        if method == 'historical':
            # Historical simulation
            var = np.percentile(returns, (1 - confidence) * 100)

        elif method == 'parametric':
            # Parametric (normal distribution)
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence)
            var = mean + z_score * std

        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion (accounts for skewness and kurtosis)
            mean = np.mean(returns)
            std = np.std(returns)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            z = stats.norm.ppf(1 - confidence)

            # Cornish-Fisher adjustment
            cf_z = (z +
                   (z**2 - 1) * skew / 6 +
                   (z**3 - 3*z) * (kurt - 3) / 24 -
                   (2*z**3 - 5*z) * skew**2 / 36)

            var = mean + cf_z * std

        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return var * 100  # Convert to percentage

    def calculate_expected_shortfall(
        self,
        df: pd.DataFrame,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        Average loss when VaR is exceeded
        More coherent risk measure than VaR

        Args:
            df: DataFrame with price data
            confidence: Confidence level

        Returns:
            Expected Shortfall as percentage (negative value)
        """
        returns = self.calculate_returns(df)
        var_threshold = np.percentile(returns, (1 - confidence) * 100)

        # Average of returns below VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return 0

        es = np.mean(tail_returns) * 100
        return es

    def calculate_downside_deviation(self, df: pd.DataFrame) -> float:
        """
        Calculate downside deviation
        Standard deviation of returns below target
        """
        returns = self.calculate_returns(df)
        downside_returns = returns[returns < self.target_return]

        if len(downside_returns) < 2:
            return 0

        downside_std = np.std(downside_returns) * np.sqrt(252) * 100
        return downside_std

    def calculate_ulcer_index(self, df: pd.DataFrame) -> float:
        """
        Calculate Ulcer Index
        Measures drawdown severity and duration
        Lower values indicate less painful drawdowns
        """
        prices = df['Close'].values
        peak = np.maximum.accumulate(prices)
        drawdown_pct = ((peak - prices) / peak) * 100

        ulcer = np.sqrt(np.mean(drawdown_pct ** 2))
        return ulcer

    def calculate_pain_index(self, df: pd.DataFrame) -> float:
        """
        Calculate Pain Index
        Mean of drawdowns (simpler than Ulcer Index)
        """
        prices = df['Close'].values
        peak = np.maximum.accumulate(prices)
        drawdown_pct = ((peak - prices) / peak) * 100

        pain = np.mean(drawdown_pct)
        return pain

    def calculate_max_drawdown_duration(self, df: pd.DataFrame) -> int:
        """
        Calculate maximum drawdown duration in days
        How long the strategy stayed in a drawdown
        """
        prices = df['Close'].values
        peak = np.maximum.accumulate(prices)

        # Find periods in drawdown
        in_drawdown = prices < peak

        if not any(in_drawdown):
            return 0

        # Calculate consecutive drawdown periods
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def calculate_skewness(self, df: pd.DataFrame) -> float:
        """Calculate return distribution skewness"""
        returns = self.calculate_returns(df)
        return stats.skew(returns)

    def calculate_kurtosis(self, df: pd.DataFrame) -> float:
        """Calculate return distribution kurtosis (excess)"""
        returns = self.calculate_returns(df)
        return stats.kurtosis(returns)

    def generate_risk_report(self, df: pd.DataFrame) -> RiskReport:
        """
        Generate comprehensive risk report

        Args:
            df: DataFrame with price data

        Returns:
            RiskReport object with all risk metrics
        """
        if len(df) < 2:
            return RiskReport(
                volatility=0, max_drawdown=0, max_drawdown_duration=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                var_95=0, var_99=0, cvar_95=0, beta=0,
                alpha=0, information_ratio=0, skewness=0, kurtosis=0
            )
        returns = self.calculate_returns(df)

        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(252) * 100

        # Calculate max drawdown
        prices = df['Close'].values
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        max_drawdown = np.max(drawdown) * 100

        # Sharpe ratio
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        years = len(df) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        sharpe = (annualized_return - self.risk_free_rate) / (volatility / 100) if volatility > 0 else 0

        return RiskReport(
            volatility=round(volatility, 2),
            max_drawdown=round(max_drawdown, 2),
            max_drawdown_duration=self.calculate_max_drawdown_duration(df),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(self.calculate_sortino_ratio(df), 2),
            calmar_ratio=round(self.calculate_calmar_ratio(df), 2),
            var_95=round(self.calculate_var(df, 0.95), 2),
            var_99=round(self.calculate_var(df, 0.99), 2),
            cvar_95=round(self.calculate_expected_shortfall(df, 0.95), 2),
            cvar_99=round(self.calculate_expected_shortfall(df, 0.99), 2),
            skewness=round(self.calculate_skewness(df), 3),
            kurtosis=round(self.calculate_kurtosis(df), 3),
            downside_deviation=round(self.calculate_downside_deviation(df), 2),
            ulcer_index=round(self.calculate_ulcer_index(df), 2),
            pain_index=round(self.calculate_pain_index(df), 2)
        )

    def risk_report_to_dict(self, report: RiskReport) -> Dict:
        """Convert RiskReport to dictionary"""
        return {
            'volatility': report.volatility,
            'max_drawdown': report.max_drawdown,
            'max_drawdown_duration': report.max_drawdown_duration,
            'sharpe_ratio': report.sharpe_ratio,
            'sortino_ratio': report.sortino_ratio,
            'calmar_ratio': report.calmar_ratio,
            'var_95': report.var_95,
            'var_99': report.var_99,
            'cvar_95': report.cvar_95,
            'cvar_99': report.cvar_99,
            'skewness': report.skewness,
            'kurtosis': report.kurtosis,
            'downside_deviation': report.downside_deviation,
            'ulcer_index': report.ulcer_index,
            'pain_index': report.pain_index
        }


def calculate_performance_metrics(df: pd.DataFrame) -> Dict:
    """
    Convenience function to calculate all performance metrics

    Args:
        df: DataFrame with price data

    Returns:
        Dictionary with all metrics
    """
    metrics = PerformanceMetrics()
    report = metrics.generate_report(df)
    return metrics.report_to_dict(report)


def calculate_risk_metrics(df: pd.DataFrame) -> Dict:
    """
    Convenience function to calculate all risk metrics

    Args:
        df: DataFrame with price data

    Returns:
        Dictionary with all risk metrics
    """
    risk = RiskMetrics()
    report = risk.generate_risk_report(df)
    return risk.risk_report_to_dict(report)


def calculate_period_returns(df: pd.DataFrame) -> Dict:
    """
    Calculate returns for different periods

    Args:
        df: DataFrame with price data

    Returns:
        Dictionary with period returns
    """
    if len(df) < 1:
        return {'current_price': 0, 'return_1d': None, 'return_5d': None,
                'return_1m': None, 'return_3m': None, 'return_6m': None, 'return_1y': None}
    current_price = float(df['Close'].iloc[-1])
    results = {'current_price': current_price}

    # Calculate returns for different periods
    periods = {
        '1d': 1,
        '5d': 5,
        '1m': 21,
        '3m': 63,
        '6m': 126,
        '1y': 252
    }

    for period_name, days in periods.items():
        if len(df) > days:
            past_price = float(df['Close'].iloc[-days-1])
            change = ((current_price - past_price) / past_price) * 100
            results[f'return_{period_name}'] = round(change, 2)
        else:
            results[f'return_{period_name}'] = None

    return results


def calculate_comprehensive_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate all performance and risk metrics in one call

    Args:
        df: DataFrame with price data

    Returns:
        Dictionary with performance and risk metrics
    """
    perf_metrics = calculate_performance_metrics(df)
    risk_metrics = calculate_risk_metrics(df)
    period_returns = calculate_period_returns(df)

    return {
        'performance': perf_metrics,
        'risk': risk_metrics,
        'returns': period_returns
    }


if __name__ == "__main__":
    # Test the performance metrics
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from data.data_loader import SaudiStockDataLoader
    from data.data_preprocessor import DataPreprocessor

    print("Testing Performance and Risk Metrics...")
    print("=" * 60)

    # Load data
    loader = SaudiStockDataLoader()
    preprocessor = DataPreprocessor()

    raw_data = loader.fetch_stock_data("2222", period="1y")
    clean_data = preprocessor.clean_data(raw_data)
    processed_data = preprocessor.add_technical_indicators(clean_data)

    print(f"Analyzing {len(processed_data)} days of data")

    # Calculate performance metrics
    perf_metrics = PerformanceMetrics()
    perf_report = perf_metrics.generate_report(processed_data)

    print(f"\n--- Performance Report ---")
    print(f"Total Return: {perf_report.total_return:.2f}%")
    print(f"Annualized Return: {perf_report.annualized_return:.2f}%")
    print(f"Volatility: {perf_report.volatility:.2f}%")
    print(f"Sharpe Ratio: {perf_report.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {perf_report.max_drawdown:.2f}%")
    print(f"Win Rate: {perf_report.win_rate:.2f}%")
    print(f"Avg Daily Return: {perf_report.avg_daily_return:.4f}%")
    print(f"Best Day: {perf_report.best_day:.2f}%")
    print(f"Worst Day: {perf_report.worst_day:.2f}%")
    print(f"Trading Days: {perf_report.trading_days}")

    # Calculate risk metrics
    risk_metrics = RiskMetrics()
    risk_report = risk_metrics.generate_risk_report(processed_data)

    print(f"\n--- Risk Report ---")
    print(f"Sortino Ratio: {risk_report.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {risk_report.calmar_ratio:.2f}")
    print(f"Max Drawdown Duration: {risk_report.max_drawdown_duration} days")
    print(f"95% VaR: {risk_report.var_95:.2f}%")
    print(f"99% VaR: {risk_report.var_99:.2f}%")
    print(f"95% Expected Shortfall: {risk_report.cvar_95:.2f}%")
    print(f"99% Expected Shortfall: {risk_report.cvar_99:.2f}%")
    print(f"Downside Deviation: {risk_report.downside_deviation:.2f}%")
    print(f"Skewness: {risk_report.skewness:.3f}")
    print(f"Kurtosis: {risk_report.kurtosis:.3f}")
    print(f"Ulcer Index: {risk_report.ulcer_index:.2f}")
    print(f"Pain Index: {risk_report.pain_index:.2f}")

    # Period returns
    period_returns = calculate_period_returns(processed_data)
    print(f"\n--- Period Returns ---")
    for key, value in period_returns.items():
        if value is not None:
            if 'price' in key:
                print(f"{key}: {value:.2f} SAR")
            else:
                print(f"{key}: {value}%")

    # Test model accuracy with dummy data
    print(f"\n--- Model Accuracy Test ---")
    predictions = np.array([100, 102, 101, 103, 105])
    actuals = np.array([100, 101, 102, 104, 104])
    accuracy = perf_metrics.calculate_model_accuracy(predictions, actuals)
    print(f"Direction Accuracy: {accuracy['direction_accuracy']:.1f}%")
    print(f"MAE: {accuracy['mae']:.4f}")
    print(f"RMSE: {accuracy['rmse']:.4f}")
    print(f"MAPE: {accuracy['mape']:.2f}%")

    # Test comprehensive metrics
    print(f"\n--- Comprehensive Metrics Test ---")
    all_metrics = calculate_comprehensive_metrics(processed_data)
    print(f"Performance keys: {list(all_metrics['performance'].keys())}")
    print(f"Risk keys: {list(all_metrics['risk'].keys())}")
    print(f"Returns keys: {list(all_metrics['returns'].keys())}")
