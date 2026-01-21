"""
Main entry point for Saudi Stock AI Analyzer
Orchestrates the complete analysis pipeline with Advanced LSTM model
"""

import sys
import os
from typing import Dict, Optional, List
from datetime import datetime
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import SaudiStockDataLoader
from data.data_preprocessor import DataPreprocessor, preprocess_for_advanced_lstm
from models.advanced_lstm import AdvancedStockLSTM, AdvancedStockPredictor, EnsemblePredictor as AdvancedEnsemble
from models.training_utils import AdvancedTrainer, create_data_loaders
from strategy.trading_strategy import TradingStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.advanced_validation import MonteCarloBacktest, AdvancedMetrics, PurgedWalkForwardCV
from backtest.performance_metrics import (
    PerformanceMetrics,
    RiskMetrics,
    calculate_performance_metrics,
    calculate_risk_metrics,
    calculate_period_returns,
    calculate_comprehensive_metrics
)
from utils.config import (
    SAUDI_STOCKS, DEFAULT_STOCK, MODEL_SAVE_PATH, SIGNAL_HISTORY_PATH,
    ADVANCED_LSTM_CONFIG, ADVANCED_LSTM_FEATURES, ADVANCED_BACKTEST_CONFIG
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
    Uses Advanced LSTM model (BiLSTM with Multi-Head Attention)
    """

    def __init__(self):
        """Initialize the stock analyzer with Advanced LSTM model"""
        self.loader = SaudiStockDataLoader()
        self.preprocessor = DataPreprocessor()
        self.strategy = TradingStrategy()
        self.metrics = PerformanceMetrics()
        self.risk_metrics = RiskMetrics()
        self.backtest_engine = BacktestEngine()
        self.signal_history = SignalHistoryManager()

        # Advanced LSTM models (lazy initialized)
        self._advanced_lstm_models = {}
        self._advanced_preprocessor = None

    def analyze_stock(
        self,
        symbol: str,
        period: str = "1y",
        train_model: bool = True,
        force_retrain: bool = False
    ) -> Dict:
        """
        Perform complete analysis on a stock using Advanced LSTM model

        Args:
            symbol: Stock symbol (e.g., '2222' for Aramco)
            period: Data period to fetch
            train_model: Whether to train/use ML model
            force_retrain: Force model retraining even if cached

        Returns:
            Dictionary with complete analysis results
        """
        print(f"Analyzing {symbol} with Advanced LSTM model...")

        # 1. Fetch data
        raw_data = self.loader.fetch_stock_data(symbol, period=period)
        if raw_data is None or raw_data.empty:
            return {'error': f'No data found for symbol {symbol}'}

        # 2. Preprocess data
        clean_data = self.preprocessor.clean_data(raw_data)
        processed_data = self.preprocessor.add_technical_indicators(clean_data)

        # 3. Get stock info
        stock_info = self._get_stock_info(symbol)

        # 4. ML prediction with Advanced LSTM
        ml_prediction = None
        ml_confidence = None
        model_info = None
        price_range = None

        if train_model and len(processed_data) > ADVANCED_LSTM_CONFIG['sequence_length'] + 50:
            try:
                ml_result = self._get_advanced_lstm_prediction(
                    symbol, processed_data, force_retrain=force_retrain
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
                'model_type': 'advanced_lstm',
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

    def _get_advanced_lstm_prediction(
        self,
        symbol: str,
        df,
        force_retrain: bool = False
    ) -> Dict:
        """
        Get prediction from Advanced LSTM model with high-accuracy architecture.

        This uses the advanced bidirectional LSTM with multi-head attention,
        residual connections, and uncertainty estimation.

        Args:
            symbol: Stock symbol
            df: Processed DataFrame
            force_retrain: Force model retraining

        Returns:
            Dictionary with prediction results including confidence
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Initialize advanced preprocessor if needed
        if self._advanced_preprocessor is None:
            self._advanced_preprocessor = DataPreprocessor()

        # Add advanced features
        df_advanced = self._advanced_preprocessor.add_technical_indicators(df.copy())
        df_advanced = self._advanced_preprocessor.add_advanced_features(df_advanced)

        # Get available features
        features = [f for f in ADVANCED_LSTM_FEATURES if f in df_advanced.columns]
        print(f"Using {len(features)} advanced features for prediction")

        config = ADVANCED_LSTM_CONFIG
        sequence_length = config['sequence_length']

        # Prepare data with proper splits
        data_splits = self._advanced_preprocessor.prepare_lstm_data_with_split(
            df_advanced, features=features,
            sequence_length=sequence_length,
            train_split=config['train_split'],
            val_split=config['val_split']
        )

        X_train = data_splits['X_train']
        X_val = data_splits['X_val']
        y_train = data_splits['y_train']
        y_val = data_splits['y_val']

        model_info = {'cached': False, 'trained': False, 'n_features': len(features)}
        model_path = os.path.join(MODEL_SAVE_PATH, f"advanced_lstm_{symbol}.pt")

        # Check for cached model
        if symbol in self._advanced_lstm_models and not force_retrain:
            predictor = self._advanced_lstm_models[symbol]
            model_info['cached'] = True
            print(f"Using cached Advanced LSTM model for {symbol}")
        elif os.path.exists(model_path) and not force_retrain:
            # Load from disk
            print(f"Loading Advanced LSTM model for {symbol} from disk...")
            predictor = AdvancedStockPredictor(
                num_features=len(features),
                hidden_sizes=config['hidden_sizes'],
                num_attention_heads=config['num_attention_heads'],
                dropout=config['dropout'],
                learning_rate=config['learning_rate']
            )
            predictor.load(model_path)
            self._advanced_lstm_models[symbol] = predictor
            model_info['cached'] = True
        else:
            # Train new model
            print(f"Training Advanced LSTM model for {symbol}...")

            # Create model
            predictor = AdvancedStockPredictor(
                num_features=len(features),
                hidden_sizes=config['hidden_sizes'],
                num_attention_heads=config['num_attention_heads'],
                dropout=config['dropout'],
                learning_rate=config['learning_rate']
            )

            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                X_train, y_train, X_val, y_val,
                batch_size=config['batch_size']
            )

            # Train with advanced trainer
            trainer = AdvancedTrainer(
                predictor.model,
                use_mixup=config.get('use_mixup', True),
                use_label_smoothing=True,
                use_ema=True,
                mixup_alpha=config.get('mixup_alpha', 0.2),
                label_smoothing=config.get('label_smoothing', 0.1)
            )

            trainer.fit(
                train_loader, val_loader,
                epochs=config['epochs'],
                patience=config['patience'],
                scheduler_type='cosine',
                verbose=True
            )

            # Save model
            predictor.save(model_path)
            self._advanced_lstm_models[symbol] = predictor
            model_info['trained'] = True
            print(f"Advanced LSTM model trained and saved for {symbol}")

        # Get prediction with uncertainty
        latest_sequence = self._advanced_preprocessor.get_latest_sequence(
            df_advanced, features=features, sequence_length=sequence_length
        )

        # Predict with MC Dropout for better uncertainty
        if config.get('output_uncertainty', True):
            result = predictor.predict_with_mc_dropout(
                latest_sequence,
                n_samples=config.get('mc_dropout_samples', 50)
            )
            prediction = result['prediction'][0][0]
            confidence = float(result['confidence'][0][0])
            uncertainty = float(result['std'][0][0])
            model_info['uncertainty'] = round(uncertainty, 4)
        else:
            result = predictor.predict(latest_sequence, return_confidence=True)
            prediction = result['prediction'][0][0]
            confidence = float(result['confidence'][0][0])

        # Determine direction
        current_price = float(df_advanced['Close'].iloc[-1])
        if len(df_advanced) >= 2:
            prev_price = float(df_advanced['Close'].iloc[-2])
            price_change = (current_price - prev_price) / prev_price
        else:
            prev_price = current_price
            price_change = 0

        # Direction based on prediction and price trend
        if prediction > 0.01 and price_change > -0.02:
            direction = "UP"
        elif prediction < -0.01 and price_change < 0.02:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        model_info['architecture'] = "BiLSTM+MultiHeadAttention+Residual"
        model_info['hidden_sizes'] = config['hidden_sizes']
        model_info['attention_heads'] = config['num_attention_heads']

        return {
            'direction': direction,
            'confidence': confidence,
            'model_info': model_info,
            'raw_prediction': float(prediction)
        }

    def run_advanced_backtest(
        self,
        symbol: str,
        period: str = "2y",
        n_simulations: int = None
    ) -> Dict:
        """
        Run advanced backtest with Monte Carlo simulation.

        Args:
            symbol: Stock symbol
            period: Data period
            n_simulations: Number of MC simulations (default from config)

        Returns:
            Advanced backtest results with probabilistic metrics
        """
        if n_simulations is None:
            n_simulations = ADVANCED_BACKTEST_CONFIG['n_monte_carlo_simulations']

        print(f"Running advanced backtest for {symbol} ({n_simulations} simulations)...")

        # Fetch and process data
        raw_data = self.loader.fetch_stock_data(symbol, period=period)
        if raw_data is None or raw_data.empty:
            return {'error': f'No data found for symbol {symbol}'}

        # Preprocess with advanced features
        df_processed, preprocessor = preprocess_for_advanced_lstm(raw_data)

        # Get features
        features = [f for f in ADVANCED_LSTM_FEATURES if f in df_processed.columns]

        # Get or train model
        prediction_result = self._get_advanced_lstm_prediction(symbol, raw_data)

        # Generate signals using the model
        sequence_length = ADVANCED_LSTM_CONFIG['sequence_length']
        signals = []
        confidences = []
        prices = df_processed['Close'].values[sequence_length:]

        # Prepare data for signal generation
        data = df_processed[features].values
        scaled_data = preprocessor.scaler.fit_transform(data)

        predictor = self._advanced_lstm_models.get(symbol)
        if predictor is None:
            return {'error': 'Model not found. Run prediction first.'}

        for i in range(sequence_length, len(scaled_data)):
            seq = scaled_data[i-sequence_length:i].reshape(1, sequence_length, len(features))
            result = predictor.predict(seq, return_confidence=True)

            pred = result['prediction'][0][0]
            conf = result['confidence'][0][0]

            if pred > 0.01 and conf > ADVANCED_BACKTEST_CONFIG['min_confidence_for_trade']:
                signals.append(1)
            elif pred < -0.01 and conf > ADVANCED_BACKTEST_CONFIG['min_confidence_for_trade']:
                signals.append(-1)
            else:
                signals.append(0)

            confidences.append(conf)

        signals = np.array(signals)
        confidences = np.array(confidences)

        # Run Monte Carlo backtest
        mc_backtest = MonteCarloBacktest(
            n_simulations=n_simulations,
            slippage_range=tuple(ADVANCED_BACKTEST_CONFIG['slippage_range']),
            commission_range=tuple(ADVANCED_BACKTEST_CONFIG['commission_range']),
            execution_delay_range=tuple(ADVANCED_BACKTEST_CONFIG['execution_delay_range']),
            use_bootstrap=ADVANCED_BACKTEST_CONFIG['use_bootstrap'],
            bootstrap_block_size=ADVANCED_BACKTEST_CONFIG['bootstrap_block_size']
        )

        mc_results = mc_backtest.run_monte_carlo(signals, prices)

        # Calculate advanced metrics on best result
        best_result = max(mc_results['results'], key=lambda r: r.metrics['sharpe_ratio'])
        advanced_metrics = AdvancedMetrics.calculate_all(
            best_result.returns,
            n_trials=n_simulations
        )

        return {
            'symbol': symbol,
            'period': period,
            'n_simulations': n_simulations,
            'monte_carlo_statistics': mc_results['statistics'],
            'best_result_metrics': best_result.metrics,
            'advanced_metrics': {
                'probabilistic_sharpe': round(advanced_metrics['probabilistic_sharpe'], 4),
                'deflated_sharpe': round(advanced_metrics['deflated_sharpe'], 4),
                'omega_ratio': round(advanced_metrics['omega_ratio'], 4),
                'tail_ratio': round(advanced_metrics['tail_ratio'], 4),
                'calmar_ratio': round(advanced_metrics['calmar_ratio'], 4),
                'sharpe_is_significant': advanced_metrics['sharpe_is_significant'],
                'strategy_is_robust': advanced_metrics['strategy_is_robust']
            },
            'signal_stats': {
                'total_signals': int(np.sum(signals != 0)),
                'buy_signals': int(np.sum(signals == 1)),
                'sell_signals': int(np.sum(signals == -1)),
                'avg_confidence': round(float(np.mean(confidences)), 4),
                'high_confidence_signals': int(np.sum(np.array(confidences) > 0.7))
            }
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
            if symbol in self._advanced_lstm_models:
                del self._advanced_lstm_models[symbol]
            print(f"Cleared cache for {symbol}")
        else:
            self._advanced_lstm_models.clear()
            print("Cleared all model caches")

    def get_available_models(self) -> List[Dict]:
        """
        Get list of available prediction models with status

        Returns:
            List with only Advanced LSTM model info
        """
        return [
            {
                'type': 'advanced_lstm',
                'name': 'Advanced LSTM (BiLSTM + Multi-Head Attention)',
                'description': 'Bidirectional LSTM with Multi-Head Attention, Residual Connections, and Uncertainty Estimation. Target: >75% accuracy, >1.5 Sharpe',
                'available': True,
                'requires_training': True,
                'recommended': True
            }
        ]

    @property
    def current_model_type(self) -> str:
        """Get current model type"""
        return 'advanced_lstm'


# Import pandas for NaN checking
import pandas as pd


def main():
    """Main function to demonstrate the analyzer"""
    print("=" * 60)
    print("Saudi Stock AI Analyzer - Advanced LSTM Edition")
    print("=" * 60)

    analyzer = StockAnalyzer()

    # Get available stocks
    stocks = analyzer.get_available_stocks()
    print(f"\nAvailable stocks: {len(stocks)}")
    for stock in stocks[:5]:
        print(f"  - {stock['symbol']}: {stock['name']} ({stock['sector']})")

    # Analyze default stock (Saudi Aramco)
    print(f"\n{'='*60}")
    print(f"Analyzing Saudi Aramco (2222) with Advanced LSTM...")
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
        print(f"\n--- ML Prediction (Advanced LSTM) ---")
        ml = result['ml_prediction']
        print(f"Direction: {ml['direction']}")
        print(f"Confidence: {ml['confidence']}%")
        print(f"Model Type: {ml['model_type']}")
        if ml['model_info']:
            if ml['model_info'].get('cached'):
                print("Model: Loaded from cache")
            elif ml['model_info'].get('trained'):
                print("Model: Newly trained")
            print(f"Architecture: {ml['model_info'].get('architecture', 'BiLSTM+Attention')}")

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
