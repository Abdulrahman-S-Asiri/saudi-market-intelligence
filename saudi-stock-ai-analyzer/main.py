"""
Main entry point for Saudi Stock AI Analyzer
Orchestrates the complete analysis pipeline with model caching
"""

import sys
import os
from typing import Dict, Optional, List
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import SaudiStockDataLoader
from data.data_preprocessor import DataPreprocessor
from models.lstm_model import StockPredictor, ModelCache
from models.ensemble_model import EnsemblePredictor, HAS_XGBOOST
from models.chronos_model import ChronosPredictor, HAS_CHRONOS, check_chronos_availability
from strategy.trading_strategy import TradingStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.performance_metrics import (
    PerformanceMetrics,
    RiskMetrics,
    calculate_performance_metrics,
    calculate_risk_metrics,
    calculate_period_returns,
    calculate_comprehensive_metrics
)
from utils.config import (
    SAUDI_STOCKS, DEFAULT_STOCK, LSTM_CONFIG, LSTM_FEATURES,
    MODEL_SAVE_PATH, SIGNAL_HISTORY_PATH, CHRONOS_CONFIG, MODEL_SELECTION
)


class SignalHistoryManager:
    """Manages signal history storage and retrieval"""

    def __init__(self, history_path: str = SIGNAL_HISTORY_PATH):
        self.history_path = history_path
        os.makedirs(history_path, exist_ok=True)

    def _get_history_file(self, symbol: str) -> str:
        return os.path.join(self.history_path, f"{symbol}_signals.json")

    def add_signal(self, symbol: str, signal_data: Dict):
        """Add a new signal to history"""
        history = self.get_history(symbol)

        # Add timestamp if not present
        if 'timestamp' not in signal_data:
            signal_data['timestamp'] = datetime.now().isoformat()

        history.append(signal_data)

        # Keep last 100 signals
        history = history[-100:]

        # Save
        with open(self._get_history_file(symbol), 'w') as f:
            json.dump(history, f, indent=2)

    def get_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get signal history for a symbol"""
        history_file = self._get_history_file(symbol)

        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                return history[-limit:]
        return []

    def update_signal_outcome(self, symbol: str, signal_idx: int, outcome: Dict):
        """Update a signal with its outcome"""
        history = self.get_history(symbol, limit=100)

        if 0 <= signal_idx < len(history):
            history[signal_idx]['outcome'] = outcome

            with open(self._get_history_file(symbol), 'w') as f:
                json.dump(history, f, indent=2)


class StockAnalyzer:
    """
    Main analyzer class that orchestrates the complete analysis pipeline
    Features model caching, ensemble predictions, and Chronos-2 forecasting
    """

    def __init__(
        self,
        use_ensemble: bool = True,
        cache_max_age: int = 86400,
        model_type: str = None
    ):
        """
        Initialize the stock analyzer

        Args:
            use_ensemble: Whether to use ensemble model (LSTM + XGBoost) - legacy param
            cache_max_age: Maximum age of cached models in seconds (default 24 hours)
            model_type: Model to use ('lstm', 'ensemble', 'chronos'). Overrides use_ensemble.
        """
        self.loader = SaudiStockDataLoader()
        self.preprocessor = DataPreprocessor()
        self.strategy = TradingStrategy()
        self.metrics = PerformanceMetrics()
        self.risk_metrics = RiskMetrics()
        self.backtest_engine = BacktestEngine()
        self.signal_history = SignalHistoryManager()

        # Model caching
        self.model_cache = ModelCache(cache_dir=MODEL_SAVE_PATH, max_age_hours=cache_max_age // 3600)
        self.ensemble_models = {}

        # Model selection
        if model_type:
            self._model_type = model_type
        elif use_ensemble and HAS_XGBOOST:
            self._model_type = "ensemble"
        else:
            self._model_type = MODEL_SELECTION.get("default_model", "lstm")

        # Legacy compatibility
        self.use_ensemble = self._model_type == "ensemble" and HAS_XGBOOST

        # Chronos predictor (lazy initialized)
        self._chronos_predictor = None

    def analyze_stock(
        self,
        symbol: str,
        period: str = "1y",
        train_model: bool = True,
        force_retrain: bool = False,
        model_type: str = None
    ) -> Dict:
        """
        Perform complete analysis on a stock

        Args:
            symbol: Stock symbol (e.g., '2222' for Aramco)
            period: Data period to fetch
            train_model: Whether to train/use ML model
            force_retrain: Force model retraining even if cached
            model_type: Override model type ('lstm', 'ensemble', 'chronos')

        Returns:
            Dictionary with complete analysis results
        """
        # Use specified model_type or fall back to instance default
        active_model_type = model_type or self._model_type
        print(f"Analyzing {symbol} with {active_model_type} model...")

        # 1. Fetch data
        raw_data = self.loader.fetch_stock_data(symbol, period=period)
        if raw_data is None or raw_data.empty:
            return {'error': f'No data found for symbol {symbol}'}

        # 2. Preprocess data
        clean_data = self.preprocessor.clean_data(raw_data)
        processed_data = self.preprocessor.add_technical_indicators(clean_data)

        # 3. Get stock info
        stock_info = self._get_stock_info(symbol)

        # 4. ML prediction (with caching)
        ml_prediction = None
        ml_confidence = None
        model_info = None
        price_range = None

        if train_model and len(processed_data) > LSTM_CONFIG['sequence_length'] + 50:
            try:
                if active_model_type == "chronos":
                    ml_result = self._get_chronos_prediction(processed_data)
                else:
                    ml_result = self._get_ml_prediction(
                        symbol, processed_data, force_retrain=force_retrain,
                        use_ensemble=(active_model_type == "ensemble")
                    )
                ml_prediction = ml_result.get('direction')
                ml_confidence = ml_result.get('confidence')
                model_info = ml_result.get('model_info')
                price_range = ml_result.get('price_range')
            except Exception as e:
                print(f"ML prediction failed: {e}")
                import traceback
                traceback.print_exc()

        # 5. Generate trading signal using enhanced strategy
        signal = self.strategy.analyze(processed_data)

        # 6. Calculate performance metrics
        performance = self.metrics.generate_report(processed_data)
        period_returns = calculate_period_returns(processed_data)

        # 7. Calculate risk metrics
        risk_report = self.risk_metrics.generate_risk_report(processed_data)

        # 8. Get current price
        current_price = float(processed_data['Close'].iloc[-1])

        # 9. Record signal to history
        signal_record = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal.action,
            'confidence': signal.confidence,
            'price': current_price,
            'indicators': {
                'rsi': signal.indicators.get('RSI', 50),
                'macd': signal.indicators.get('MACD', 0),
                'regime': signal.market_regime
            }
        }
        self.signal_history.add_signal(symbol, signal_record)

        # Compile results
        return {
            'symbol': symbol,
            'name': stock_info.get('name', symbol),
            'sector': stock_info.get('sector', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'signal': {
                'action': signal.action,
                'confidence': round(signal.confidence, 1),
                'reasons': signal.reasons,
                'price': round(current_price, 2),
                'stop_loss': round(signal.stop_loss, 2) if signal.stop_loss else None,
                'take_profit': round(signal.take_profit, 2) if signal.take_profit else None,
                'market_regime': signal.market_regime
            },
            'ml_prediction': {
                'direction': ml_prediction,
                'confidence': round(ml_confidence * 100, 1) if ml_confidence else None,
                'model_type': active_model_type,
                'model_info': model_info,
                'price_range': price_range
            } if ml_prediction else None,
            'indicators': {
                'rsi': round(signal.indicators.get('RSI', 50), 2),
                'macd': round(signal.indicators.get('MACD', 0), 4),
                'macd_signal': round(signal.indicators.get('MACD_Signal', 0), 4),
                'sma_20': round(signal.indicators.get('SMA_20', current_price), 2),
                'sma_50': round(signal.indicators.get('SMA_50', current_price), 2),
                'atr': round(signal.indicators.get('ATR', 0), 4),
                'adx': round(signal.indicators.get('ADX', 0), 2),
                'stochastic_k': round(signal.indicators.get('Stochastic_K', 50), 2),
                'williams_r': round(signal.indicators.get('Williams_R', -50), 2)
            },
            'trend': {
                'direction': signal.market_regime,
                'strength': round(signal.trend_strength * 100, 1) if hasattr(signal, 'trend_strength') else 50
            },
            'performance': self.metrics.report_to_dict(performance),
            'risk': self.risk_metrics.risk_report_to_dict(risk_report),
            'period_returns': period_returns,
            'data_points': len(processed_data)
        }

    def _get_stock_info(self, symbol: str) -> Dict:
        """Get stock metadata"""
        if symbol in SAUDI_STOCKS:
            return SAUDI_STOCKS[symbol]
        return {'name': symbol, 'sector': 'Unknown'}

    def _get_ml_prediction(
        self,
        symbol: str,
        df,
        force_retrain: bool = False,
        use_ensemble: bool = None
    ) -> Dict:
        """
        Get ML prediction for stock with caching

        Args:
            symbol: Stock symbol
            df: Processed DataFrame
            force_retrain: Force model retraining
            use_ensemble: Whether to use ensemble model (defaults to instance setting)

        Returns:
            Dictionary with prediction results
        """
        # Use instance setting if not specified
        if use_ensemble is None:
            use_ensemble = self.use_ensemble
        features = [f for f in LSTM_FEATURES if f in df.columns]

        # Prepare data with proper train/val/test split
        data_splits = self.preprocessor.prepare_lstm_data_with_split(
            df, features=features
        )

        X_train = data_splits['X_train']
        X_val = data_splits['X_val']
        X_test = data_splits['X_test']
        y_train = data_splits['y_train']
        y_val = data_splits['y_val']
        y_test = data_splits['y_test']

        model_info = {'cached': False, 'trained': False}

        if use_ensemble and HAS_XGBOOST:
            # Use ensemble model
            if symbol not in self.ensemble_models:
                self.ensemble_models[symbol] = EnsemblePredictor(input_size=len(features))

            ensemble = self.ensemble_models[symbol]

            # Try to load cached model
            if not force_retrain and ensemble.load(symbol):
                model_info['cached'] = True
                print(f"Loaded cached ensemble model for {symbol}")
            else:
                # Train new model
                print(f"Training ensemble model for {symbol}...")
                ensemble.train(
                    X_train, y_train, X_val, y_val,
                    epochs=LSTM_CONFIG.get('epochs', 100),
                    patience=LSTM_CONFIG.get('patience', 10),
                    verbose=False
                )
                ensemble.save(symbol)
                model_info['trained'] = True
                print(f"Saved ensemble model for {symbol}")

            # Get prediction
            latest_sequence = self.preprocessor.get_latest_sequence(df, features=features)
            current_price = float(df['Close'].iloc[-1])

            direction, confidence, breakdown = ensemble.predict_direction(
                latest_sequence, current_price
            )

            model_info['lstm_weight'] = round(ensemble.lstm_weight, 2)
            model_info['xgb_weight'] = round(ensemble.xgb_weight, 2)

            return {
                'direction': direction,
                'confidence': confidence,
                'model_info': model_info
            }

        else:
            # Use LSTM only with caching
            predictor = self.model_cache.get(symbol)

            if predictor is None or force_retrain:
                # Create and train new model
                predictor = StockPredictor(
                    input_size=len(features),
                    use_attention=LSTM_CONFIG.get('use_attention', True)
                )

                print(f"Training LSTM model for {symbol}...")
                predictor.train(
                    X_train, y_train, X_val, y_val,
                    epochs=LSTM_CONFIG.get('epochs', 100),
                    patience=LSTM_CONFIG.get('patience', 10),
                    verbose=False
                )

                # Cache the model
                self.model_cache.set(symbol, predictor)
                model_info['trained'] = True
                print(f"Cached LSTM model for {symbol}")
            else:
                model_info['cached'] = True
                print(f"Using cached LSTM model for {symbol}")

            # Get prediction
            latest_sequence = self.preprocessor.get_latest_sequence(df, features=features)
            current_scaled = latest_sequence[0, -1, 0]

            direction, confidence = predictor.predict_direction(latest_sequence, current_scaled)

            return {
                'direction': direction,
                'confidence': confidence,
                'model_info': model_info
            }

    def _get_chronos_prediction(self, df) -> Dict:
        """
        Get Chronos-2 prediction for stock

        Args:
            df: Processed DataFrame with OHLCV data

        Returns:
            Dictionary with prediction results including confidence intervals
        """
        if not HAS_CHRONOS:
            raise ImportError(
                "Chronos not installed. Install with: pip install chronos-forecasting"
            )

        # Initialize Chronos predictor if needed
        if self._chronos_predictor is None:
            self._chronos_predictor = ChronosPredictor(
                model_name=CHRONOS_CONFIG.get("model_name", "amazon/chronos-t5-small"),
                prediction_length=CHRONOS_CONFIG.get("prediction_length", 5),
                context_length=CHRONOS_CONFIG.get("context_length", 60),
                quantile_levels=CHRONOS_CONFIG.get("quantile_levels", [0.1, 0.5, 0.9]),
                device=CHRONOS_CONFIG.get("device", "cpu")
            )

        # Get prediction
        direction, confidence, details = self._chronos_predictor.predict_direction(df)

        model_info = {
            "model": "chronos",
            "model_name": CHRONOS_CONFIG.get("model_name"),
            "prediction_length": CHRONOS_CONFIG.get("prediction_length"),
            "cached": self._chronos_predictor.is_loaded
        }

        return {
            'direction': direction,
            'confidence': confidence,
            'model_info': model_info,
            'price_range': details.get('price_range'),
            'forecast_path': details.get('forecast_path')
        }

    def run_backtest(
        self,
        symbol: str,
        period: str = "1y",
        min_confidence: float = 60,
        hold_period: int = 5
    ) -> Dict:
        """
        Run backtest on a stock

        Args:
            symbol: Stock symbol
            period: Data period
            min_confidence: Minimum confidence to take trades
            hold_period: Default holding period in days

        Returns:
            Backtest results dictionary
        """
        print(f"Running backtest for {symbol}...")

        # Fetch and process data
        raw_data = self.loader.fetch_stock_data(symbol, period=period)
        if raw_data is None or raw_data.empty:
            return {'error': f'No data found for symbol {symbol}'}

        clean_data = self.preprocessor.clean_data(raw_data)
        processed_data = self.preprocessor.add_technical_indicators(clean_data)

        # Run backtest
        result = self.backtest_engine.run(
            processed_data,
            min_confidence=min_confidence,
            hold_period=hold_period
        )

        # Run signal accuracy test
        signal_accuracy = self.backtest_engine.run_signal_backtest(processed_data)

        return {
            'symbol': symbol,
            'period': period,
            'backtest': self.backtest_engine.result_to_dict(result),
            'signal_accuracy': signal_accuracy
        }

    def get_signal_history(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Get signal history for a stock"""
        return self.signal_history.get_history(symbol, limit=limit)

    def get_chart_data(self, symbol: str, period: str = "6mo") -> Dict:
        """
        Get chart data for frontend visualization

        Args:
            symbol: Stock symbol
            period: Data period

        Returns:
            Dictionary with chart-ready data
        """
        raw_data = self.loader.fetch_stock_data(symbol, period=period)
        if raw_data is None or raw_data.empty:
            return {'error': f'No data found for symbol {symbol}'}

        clean_data = self.preprocessor.clean_data(raw_data)
        processed_data = self.preprocessor.add_technical_indicators(clean_data)

        # Convert to list of dicts for JSON serialization
        chart_data = []
        for idx, row in processed_data.iterrows():
            entry = {
                'date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume']),
            }

            # Add indicators if available
            if 'SMA_20' in row and not pd.isna(row['SMA_20']):
                entry['sma_20'] = round(float(row['SMA_20']), 2)
            if 'SMA_50' in row and not pd.isna(row['SMA_50']):
                entry['sma_50'] = round(float(row['SMA_50']), 2)
            if 'RSI' in row and not pd.isna(row['RSI']):
                entry['rsi'] = round(float(row['RSI']), 2)
            if 'MACD' in row and not pd.isna(row['MACD']):
                entry['macd'] = round(float(row['MACD']), 4)
            if 'MACD_Signal' in row and not pd.isna(row['MACD_Signal']):
                entry['macd_signal'] = round(float(row['MACD_Signal']), 4)
            if 'ATR' in row and not pd.isna(row['ATR']):
                entry['atr'] = round(float(row['ATR']), 4)

            chart_data.append(entry)

        return {
            'symbol': symbol,
            'data': chart_data,
            'count': len(chart_data)
        }

    def compare_stocks(self, symbols: List[str], period: str = "6mo") -> Dict:
        """
        Compare multiple stocks

        Args:
            symbols: List of stock symbols
            period: Data period

        Returns:
            Comparison results
        """
        results = {}

        for symbol in symbols:
            analysis = self.analyze_stock(symbol, period=period, train_model=False)
            if 'error' not in analysis:
                results[symbol] = {
                    'name': analysis['name'],
                    'signal': analysis['signal']['action'],
                    'confidence': analysis['signal']['confidence'],
                    'price': analysis['signal']['price'],
                    'total_return': analysis['performance']['total_return'],
                    'volatility': analysis['performance']['volatility'],
                    'sharpe_ratio': analysis['performance']['sharpe_ratio'],
                    'rsi': analysis['indicators']['rsi']
                }

        return {
            'comparison': results,
            'period': period,
            'timestamp': datetime.now().isoformat()
        }

    def get_available_stocks(self) -> list:
        """Get list of available stocks"""
        return [
            {
                'symbol': symbol,
                'name': info['name'],
                'sector': info['sector']
            }
            for symbol, info in SAUDI_STOCKS.items()
        ]

    def clear_model_cache(self, symbol: Optional[str] = None):
        """Clear cached models"""
        if symbol:
            self.model_cache.invalidate(symbol)
            if symbol in self.ensemble_models:
                del self.ensemble_models[symbol]
            print(f"Cleared cache for {symbol}")
        else:
            self.model_cache.clear()
            self.ensemble_models.clear()
            print("Cleared all model caches")

    def set_model_type(self, model_type: str) -> Dict:
        """
        Set the active model type for predictions

        Args:
            model_type: Model type ('lstm', 'ensemble', 'chronos')

        Returns:
            Dictionary with status and model info
        """
        available = self.get_available_models()
        available_types = [m['type'] for m in available if m['available']]

        if model_type not in MODEL_SELECTION['available_models']:
            return {
                'success': False,
                'error': f"Unknown model type: {model_type}",
                'available_models': available_types
            }

        if model_type == 'chronos' and not HAS_CHRONOS:
            return {
                'success': False,
                'error': "Chronos not installed. Install with: pip install chronos-forecasting",
                'available_models': available_types
            }

        if model_type == 'ensemble' and not HAS_XGBOOST:
            return {
                'success': False,
                'error': "XGBoost not installed. Install with: pip install xgboost",
                'available_models': available_types
            }

        self._model_type = model_type
        self.use_ensemble = (model_type == "ensemble")

        return {
            'success': True,
            'model_type': model_type,
            'message': f"Model switched to {model_type}"
        }

    def get_available_models(self) -> List[Dict]:
        """
        Get list of available prediction models with status

        Returns:
            List of model info dictionaries
        """
        return [
            {
                'type': 'lstm',
                'name': 'LSTM Neural Network',
                'description': 'Deep learning model for sequence prediction',
                'available': True,
                'requires_training': True
            },
            {
                'type': 'ensemble',
                'name': 'Ensemble (LSTM + XGBoost)',
                'description': 'Combined deep learning and gradient boosting',
                'available': HAS_XGBOOST,
                'requires_training': True,
                'note': None if HAS_XGBOOST else 'Install xgboost: pip install xgboost'
            },
            {
                'type': 'chronos',
                'name': 'Chronos-2 Foundation Model',
                'description': 'Amazon\'s zero-shot time series forecaster with confidence intervals',
                'available': HAS_CHRONOS,
                'requires_training': False,
                'note': None if HAS_CHRONOS else 'Install chronos: pip install chronos-forecasting'
            }
        ]

    @property
    def current_model_type(self) -> str:
        """Get current model type"""
        return self._model_type


# Import pandas for NaN checking
import pandas as pd


def main():
    """Main function to demonstrate the analyzer"""
    print("=" * 60)
    print("Saudi Stock AI Analyzer - Demo")
    print("=" * 60)

    analyzer = StockAnalyzer(use_ensemble=True)

    # Get available stocks
    stocks = analyzer.get_available_stocks()
    print(f"\nAvailable stocks: {len(stocks)}")
    for stock in stocks[:5]:
        print(f"  - {stock['symbol']}: {stock['name']} ({stock['sector']})")

    # Analyze default stock (Saudi Aramco)
    print(f"\n{'='*60}")
    print(f"Analyzing Saudi Aramco (2222)...")
    print("=" * 60)

    result = analyzer.analyze_stock("2222", period="6mo", train_model=True)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    # Display results
    print(f"\n--- Stock Info ---")
    print(f"Symbol: {result['symbol']}")
    print(f"Name: {result['name']}")
    print(f"Sector: {result['sector']}")

    print(f"\n--- Trading Signal ---")
    signal = result['signal']
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']}%")
    print(f"Current Price: {signal['price']} SAR")
    print(f"Market Regime: {signal['market_regime']}")
    if signal['stop_loss']:
        print(f"Stop Loss: {signal['stop_loss']} SAR")
    if signal['take_profit']:
        print(f"Take Profit: {signal['take_profit']} SAR")
    print("Reasons:")
    for reason in signal['reasons']:
        print(f"  - {reason}")

    if result['ml_prediction']:
        print(f"\n--- ML Prediction ---")
        ml = result['ml_prediction']
        print(f"Direction: {ml['direction']}")
        print(f"Confidence: {ml['confidence']}%")
        print(f"Model Type: {ml['model_type']}")
        if ml['model_info']:
            if ml['model_info'].get('cached'):
                print("Model: Loaded from cache")
            elif ml['model_info'].get('trained'):
                print("Model: Newly trained")
            if 'lstm_weight' in ml['model_info']:
                print(f"LSTM Weight: {ml['model_info']['lstm_weight']}")
                print(f"XGBoost Weight: {ml['model_info']['xgb_weight']}")

    print(f"\n--- Technical Indicators ---")
    ind = result['indicators']
    print(f"RSI: {ind['rsi']}")
    print(f"MACD: {ind['macd']}")
    print(f"SMA 20: {ind['sma_20']}")
    print(f"SMA 50: {ind['sma_50']}")
    print(f"ATR: {ind['atr']}")
    print(f"ADX: {ind['adx']}")

    print(f"\n--- Performance Metrics ---")
    perf = result['performance']
    print(f"Total Return: {perf['total_return']}%")
    print(f"Volatility: {perf['volatility']}%")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']}")
    print(f"Max Drawdown: {perf['max_drawdown']}%")
    print(f"Win Rate: {perf['win_rate']}%")

    print(f"\n--- Risk Metrics ---")
    risk = result['risk']
    print(f"Sortino Ratio: {risk['sortino_ratio']}")
    print(f"Calmar Ratio: {risk['calmar_ratio']}")
    print(f"95% VaR: {risk['var_95']}%")
    print(f"95% Expected Shortfall: {risk['cvar_95']}%")

    # Run backtest
    print(f"\n{'='*60}")
    print("Running Backtest...")
    print("=" * 60)

    backtest = analyzer.run_backtest("2222", period="1y")
    if 'error' not in backtest:
        bt = backtest['backtest']
        print(f"Total Trades: {bt['total_trades']}")
        print(f"Win Rate: {bt['win_rate']:.1f}%")
        print(f"Profit Factor: {bt['profit_factor']:.2f}")
        print(f"Total Return: {bt['total_return']}%")
        print(f"Max Drawdown: {bt['max_drawdown']}%")

        sa = backtest['signal_accuracy']
        print(f"\nSignal Accuracy: {sa['accuracy']}%")
        print(f"High Confidence Accuracy: {sa['high_confidence_accuracy']}%")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
