#!/usr/bin/env python3
# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
Saudi Market Scanner - Automated AI Analysis
=============================================

Scans top Saudi stocks using trained TFT models to generate
trading signals with confidence scores.

Features:
- Auto-trains models if no checkpoint exists
- Generates 5-day forecasts with uncertainty bands
- Calculates confidence scores from prediction intervals
- Produces actionable signals: STRONG BUY, BUY, HOLD, SELL
- Outputs clean CSV report

Usage:
    python market_scanner.py
    python market_scanner.py --tickers 2222 1120 2010
    python market_scanner.py --train_epochs 10 --output_dir ./reports

Author: Claude AI / Abdulrahman Asiri
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_tft import SOTAStockPredictor, TFTConfig
from data.data_loader_tft import (
    create_tft_dataloaders,
    TFTFeatureConfig,
    pad_or_trim_features,
    get_sector_id,
    get_market_cap_group
)
from data.data_preprocessor import preprocess_stock_data
from data.data_loader import SaudiStockDataLoader
from train_main import TFTTrainingConfig, TFTMainTrainer
from utils.config import MODEL_SAVE_PATH, SAUDI_STOCKS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm not installed. Install with: pip install tqdm")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Top 10 Saudi Stocks for scanning
DEFAULT_TICKERS = [
    '2222',  # Saudi Aramco (Energy)
    '1120',  # Al Rajhi Bank (Banks)
    '2010',  # SABIC (Materials)
    '1180',  # Al Inma Bank (Banks)
    '1182',  # Saudi National Bank (Banks)
    '2310',  # Sipchem (Materials)
    '7010',  # STC (Telecom)
    '4030',  # Bahri (Transportation)
    '2280',  # Almarai (Food)
    '1010',  # Riyad Bank (Banks)
]

# Signal thresholds
SIGNAL_THRESHOLDS = {
    'strong_buy_return': 0.015,      # 1.5% expected return
    'buy_return': 0.01,              # 1.0% expected return
    'sell_return': -0.005,           # -0.5% expected return
    'high_confidence': 0.7,          # Confidence threshold for STRONG signals
    'medium_confidence': 0.4,        # Confidence threshold for regular signals
}


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """
    Generates trading signals from model predictions.

    Signal Logic:
    - STRONG BUY: Return > 1.5% AND High Confidence
    - BUY: Return > 1.0% AND Medium Confidence
    - HOLD: Mixed signals or uncertain
    - SELL: Return < -0.5%
    - STRONG SELL: Return < -1.5% AND High Confidence
    """

    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or SIGNAL_THRESHOLDS

    def calculate_confidence(
        self,
        lower_quantile: float,
        upper_quantile: float,
        normalize: bool = True
    ) -> float:
        """
        Calculate confidence score from prediction interval.

        Confidence = 1 / (Upper - Lower)
        Narrower bands = Higher confidence

        Args:
            lower_quantile: 10th percentile prediction
            upper_quantile: 90th percentile prediction
            normalize: Whether to normalize to 0-1 range

        Returns:
            Confidence score (0-1 if normalized)
        """
        interval_width = abs(upper_quantile - lower_quantile)

        # Avoid division by zero
        if interval_width < 1e-6:
            return 1.0

        # Raw confidence (inverse of width)
        raw_confidence = 1.0 / (1.0 + interval_width * 10)  # Scale factor

        if normalize:
            # Clip to 0-1 range
            return min(max(raw_confidence, 0.0), 1.0)

        return raw_confidence

    def generate_signal(
        self,
        expected_return: float,
        confidence: float
    ) -> Tuple[str, str]:
        """
        Generate trading signal based on expected return and confidence.

        Args:
            expected_return: Predicted return (e.g., 0.02 = 2%)
            confidence: Confidence score (0-1)

        Returns:
            Tuple of (signal_code, signal_description)
        """
        t = self.thresholds

        # STRONG BUY: High return + High confidence
        if expected_return > t['strong_buy_return'] and confidence >= t['high_confidence']:
            return 'STRONG_BUY', '🟢🟢 STRONG BUY - High confidence bullish'

        # BUY: Positive return + Medium confidence
        elif expected_return > t['buy_return'] and confidence >= t['medium_confidence']:
            return 'BUY', '🟢 BUY - Bullish signal'

        # STRONG SELL: Large negative return + High confidence
        elif expected_return < -t['strong_buy_return'] and confidence >= t['high_confidence']:
            return 'STRONG_SELL', '🔴🔴 STRONG SELL - High confidence bearish'

        # SELL: Negative return
        elif expected_return < t['sell_return']:
            return 'SELL', '🔴 SELL - Bearish signal'

        # HOLD: Mixed or uncertain
        else:
            if confidence < t['medium_confidence']:
                return 'HOLD', '🟡 HOLD - High uncertainty'
            else:
                return 'HOLD', '🟡 HOLD - Neutral outlook'


# =============================================================================
# MARKET SCANNER
# =============================================================================

class SaudiMarketScanner:
    """
    Automated market scanner for Saudi stocks.

    Scans multiple stocks, generates predictions, and produces
    a consolidated report with trading signals.
    """

    def __init__(
        self,
        tickers: List[str] = None,
        model_dir: str = None,
        device: str = None,
        train_epochs: int = 5,
        data_period: str = '2y'
    ):
        """
        Initialize the market scanner.

        Args:
            tickers: List of stock tickers to scan
            model_dir: Directory for model checkpoints
            device: Device for inference (cuda/cpu)
            train_epochs: Epochs for quick training if no model exists
            data_period: Historical data period
        """
        self.tickers = tickers or DEFAULT_TICKERS
        self.model_dir = model_dir or MODEL_SAVE_PATH
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_epochs = train_epochs
        self.data_period = data_period

        # Initialize components
        self.data_loader = SaudiStockDataLoader()
        self.signal_generator = SignalGenerator()

        # Results storage
        self.results = []
        self.errors = []

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        logger.info(f"Initialized SaudiMarketScanner")
        logger.info(f"  Tickers: {len(self.tickers)}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model directory: {self.model_dir}")

    def _get_checkpoint_path(self, ticker: str) -> str:
        """Get checkpoint path for a ticker."""
        return os.path.join(self.model_dir, f'tft_{ticker}_best.pt')

    def _model_exists(self, ticker: str) -> bool:
        """Check if a trained model exists for the ticker."""
        return os.path.exists(self._get_checkpoint_path(ticker))

    def _load_model(self, ticker: str) -> Optional[SOTAStockPredictor]:
        """Load a trained model for the ticker."""
        checkpoint_path = self._get_checkpoint_path(ticker)

        if not os.path.exists(checkpoint_path):
            return None

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False
            )

            config_dict = checkpoint.get('config', {})

            model_config = TFTConfig(
                num_static_features=2,
                num_dynamic_features=config_dict.get('num_dynamic_features', 40),
                num_known_features=config_dict.get('num_known_features', 5),
                hidden_size=config_dict.get('hidden_size', 128),
                lstm_layers=config_dict.get('lstm_layers', 2),
                attention_heads=config_dict.get('attention_heads', 4),
                dropout=0.0,  # No dropout for inference
                sequence_length=config_dict.get('sequence_length', 60),
                forecast_horizon=config_dict.get('forecast_horizon', 5),
                quantiles=tuple(config_dict.get('quantiles', (0.1, 0.5, 0.9)))
            )

            model = SOTAStockPredictor(model_config, device=self.device)
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            return model

        except Exception as e:
            logger.error(f"Failed to load model for {ticker}: {e}")
            return None

    def _train_model(self, ticker: str, df: pd.DataFrame) -> Optional[str]:
        """
        Quick-train a model for the ticker.

        Args:
            ticker: Stock ticker
            df: Preprocessed DataFrame

        Returns:
            Path to saved checkpoint or None if failed
        """
        logger.info(f"  Training new model for {ticker} ({self.train_epochs} epochs)...")

        try:
            # Estimate market cap
            avg_price = df['Close'].mean() if 'Close' in df.columns else 100
            market_cap = avg_price * 1e9  # Rough estimate

            # Create DataLoaders
            feature_config = TFTFeatureConfig(
                sequence_length=60,
                forecast_horizon=5
            )

            train_loader, val_loader, _, metadata = create_tft_dataloaders(
                df=df,
                stock_code=ticker,
                market_cap=market_cap,
                feature_config=feature_config,
                train_split=0.8,
                val_split=0.15,
                batch_size=32
            )

            if len(train_loader) == 0:
                logger.warning(f"  Not enough data to train model for {ticker}")
                return None

            # Create training config
            training_config = TFTTrainingConfig(
                hidden_size=64,  # Smaller for quick training
                lstm_layers=1,
                attention_heads=2,
                num_dynamic_features=max(40, metadata['num_observed_features']),
                num_known_features=max(5, metadata['num_known_features']),
                epochs=self.train_epochs,
                batch_size=32,
                learning_rate=1e-3,
                patience=self.train_epochs,  # No early stopping for quick train
                checkpoint_dir=self.model_dir,
            )

            # Train
            trainer = TFTMainTrainer(config=training_config, device=self.device)
            trainer.fit(train_loader, val_loader, metadata=metadata)

            # Copy best checkpoint to ticker-specific name
            src_path = os.path.join(self.model_dir, 'sota_tft_best_sharpe.pt')
            dst_path = self._get_checkpoint_path(ticker)

            if os.path.exists(src_path):
                import shutil
                shutil.copy(src_path, dst_path)
                return dst_path

            return None

        except Exception as e:
            logger.error(f"  Training failed for {ticker}: {e}")
            return None

    @torch.no_grad()
    def _predict(
        self,
        model: SOTAStockPredictor,
        df: pd.DataFrame,
        ticker: str
    ) -> Optional[Dict]:
        """
        Generate predictions for a stock.

        Args:
            model: Trained TFT model
            df: Preprocessed DataFrame
            ticker: Stock ticker

        Returns:
            Dictionary with predictions and metrics
        """
        model.eval()
        config = model.config

        # Get the latest sequence
        seq_len = config.sequence_length

        if len(df) < seq_len + 1:
            return None

        # Prepare input data
        feature_config = TFTFeatureConfig(
            sequence_length=seq_len,
            forecast_horizon=config.forecast_horizon
        )

        # Get available features
        observed_features = [f for f in feature_config.observed_features if f in df.columns]
        known_features = [f for f in feature_config.known_features if f in df.columns]

        # Add calendar features if missing
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            df.index = pd.to_datetime(df.index)

        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df.index.dayofweek / 6.0
        if 'day_of_month' not in df.columns:
            df['day_of_month'] = df.index.day / 31.0
        if 'month' not in df.columns:
            df['month'] = df.index.month / 12.0
        if 'quarter' not in df.columns:
            df['quarter'] = df.index.quarter / 4.0
        if 'is_month_end' not in df.columns:
            df['is_month_end'] = df.index.is_month_end.astype(float)

        # Re-get available features
        observed_features = [f for f in feature_config.observed_features if f in df.columns]
        known_features = [f for f in feature_config.known_features if f in df.columns]

        # Extract latest sequence
        latest_data = df.tail(seq_len)

        # Observed features
        x_observed = latest_data[observed_features].values
        x_observed = np.nan_to_num(x_observed, nan=0.0, posinf=0.0, neginf=0.0)

        # Known features
        x_known = latest_data[known_features].values
        x_known = np.nan_to_num(x_known, nan=0.0, posinf=0.0, neginf=0.0)

        # Static features
        sector_id = get_sector_id(ticker)
        market_cap_group = 0  # Default to large cap
        x_static = np.array([[sector_id, market_cap_group]], dtype=np.float32)

        # Convert to tensors
        x_observed = torch.tensor(x_observed, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_known = torch.tensor(x_known, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_static = torch.tensor(x_static, dtype=torch.float32).to(self.device)

        # Pad/trim to model dimensions
        x_observed, x_known = pad_or_trim_features(
            x_observed, x_known,
            target_observed=config.num_dynamic_features,
            target_known=config.num_known_features
        )

        # Forward pass
        output = model(
            static_features=x_static,
            dynamic_features=x_observed,
            known_features=x_known,
            return_attention=False
        )

        # Extract predictions
        predictions = output['quantile_predictions'].cpu().numpy()[0]  # (horizon, quantiles)

        # Get current price
        current_price = df['Close'].iloc[-1]

        # Predictions are returns, convert to prices
        pred_returns = predictions[:, 1]  # Median predictions
        pred_low = predictions[:, 0]      # 10th percentile
        pred_high = predictions[:, 2]     # 90th percentile

        # 5-day prediction
        pred_return_5d = pred_returns[4] if len(pred_returns) >= 5 else pred_returns[-1]
        pred_low_5d = pred_low[4] if len(pred_low) >= 5 else pred_low[-1]
        pred_high_5d = pred_high[4] if len(pred_high) >= 5 else pred_high[-1]

        # Convert return to price
        pred_price_5d = current_price * (1 + pred_return_5d)

        # Calculate confidence
        confidence = self.signal_generator.calculate_confidence(pred_low_5d, pred_high_5d)

        return {
            'current_price': current_price,
            'pred_price_5d': pred_price_5d,
            'pred_return_5d': pred_return_5d,
            'pred_low_5d': pred_low_5d,
            'pred_high_5d': pred_high_5d,
            'confidence': confidence,
            'predictions': predictions
        }

    def scan_stock(self, ticker: str) -> Optional[Dict]:
        """
        Scan a single stock.

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with scan results or None if failed
        """
        try:
            stock_info = SAUDI_STOCKS.get(ticker, {})
            stock_name = stock_info.get('name', 'Unknown')
            sector = stock_info.get('sector', 'Unknown')

            logger.info(f"Scanning {ticker} ({stock_name})...")

            # Step A: Load data
            try:
                df = self.data_loader.fetch_stock_with_macro(ticker, period=self.data_period)
            except Exception:
                df = self.data_loader.fetch_stock_data(ticker, period=self.data_period)

            if df is None or len(df) < 100:
                logger.warning(f"  Insufficient data for {ticker}")
                return None

            # Preprocess
            processed_df = preprocess_stock_data(df, include_advanced=True, include_macro=True)

            if len(processed_df) < 70:
                logger.warning(f"  Not enough processed data for {ticker}")
                return None

            # Step B: Check for model or train
            model = self._load_model(ticker)

            if model is None:
                logger.info(f"  No model found. Training...")
                checkpoint_path = self._train_model(ticker, processed_df)

                if checkpoint_path:
                    model = self._load_model(ticker)

            if model is None:
                logger.error(f"  Could not load/train model for {ticker}")
                return None

            # Step C: Generate predictions
            predictions = self._predict(model, processed_df, ticker)

            if predictions is None:
                logger.error(f"  Prediction failed for {ticker}")
                return None

            # Step D: Generate signal
            signal_code, signal_desc = self.signal_generator.generate_signal(
                predictions['pred_return_5d'],
                predictions['confidence']
            )

            # Compile results
            result = {
                'Ticker': ticker,
                'Name': stock_name,
                'Sector': sector,
                'Close_Price': round(predictions['current_price'], 2),
                'Pred_Price_5D': round(predictions['pred_price_5d'], 2),
                'Exp_Return': round(predictions['pred_return_5d'] * 100, 2),  # As percentage
                'Pred_Low': round(predictions['pred_low_5d'] * 100, 2),
                'Pred_High': round(predictions['pred_high_5d'] * 100, 2),
                'Confidence': round(predictions['confidence'] * 100, 1),  # As percentage
                'AI_Signal': signal_code,
                'Signal_Desc': signal_desc,
                'Scan_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.info(f"  ✓ {ticker}: {signal_code} | Return: {result['Exp_Return']:.1f}% | Conf: {result['Confidence']:.0f}%")

            return result

        except Exception as e:
            logger.error(f"  ✗ Error scanning {ticker}: {str(e)}")
            self.errors.append({'ticker': ticker, 'error': str(e)})
            return None

    def run_scan(self) -> pd.DataFrame:
        """
        Run the full market scan.

        Returns:
            DataFrame with scan results
        """
        logger.info("=" * 60)
        logger.info("Saudi Market Scanner - Starting Full Scan")
        logger.info("=" * 60)
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Tickers to scan: {len(self.tickers)}")
        logger.info("")

        self.results = []
        self.errors = []

        # Progress bar
        if HAS_TQDM:
            iterator = tqdm(self.tickers, desc="Scanning stocks", unit="stock")
        else:
            iterator = self.tickers

        for ticker in iterator:
            result = self.scan_stock(ticker)
            if result:
                self.results.append(result)

        # Create DataFrame
        if self.results:
            df_results = pd.DataFrame(self.results)

            # Sort by expected return
            df_results = df_results.sort_values('Exp_Return', ascending=False)

        else:
            df_results = pd.DataFrame()

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Scan Complete!")
        logger.info(f"  Successful: {len(self.results)}/{len(self.tickers)}")
        logger.info(f"  Failed: {len(self.errors)}")

        if self.results:
            # Signal distribution
            signals = df_results['AI_Signal'].value_counts()
            logger.info(f"\nSignal Distribution:")
            for signal, count in signals.items():
                logger.info(f"  {signal}: {count}")

            # Top picks
            top_picks = df_results[df_results['AI_Signal'].isin(['STRONG_BUY', 'BUY'])].head(3)
            if not top_picks.empty:
                logger.info(f"\nTop Picks:")
                for _, row in top_picks.iterrows():
                    logger.info(f"  {row['Ticker']} ({row['Name']}): {row['AI_Signal']} | {row['Exp_Return']:.1f}%")

        logger.info("=" * 60)

        return df_results

    def save_report(
        self,
        df_results: pd.DataFrame,
        output_dir: str = None,
        filename: str = None
    ) -> str:
        """
        Save scan results to CSV.

        Args:
            df_results: Results DataFrame
            output_dir: Output directory
            filename: Custom filename

        Returns:
            Path to saved report
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'reports')

        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f'saudi_market_report_{date_str}.csv'

        filepath = os.path.join(output_dir, filename)

        # Select columns for clean report
        report_columns = [
            'Ticker', 'Name', 'Sector', 'Close_Price', 'Pred_Price_5D',
            'Exp_Return', 'Confidence', 'AI_Signal', 'Signal_Desc'
        ]

        available_columns = [c for c in report_columns if c in df_results.columns]

        df_results[available_columns].to_csv(filepath, index=False)

        logger.info(f"\nReport saved to: {filepath}")

        return filepath


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Saudi Market Scanner - AI-Powered Stock Analysis'
    )

    parser.add_argument(
        '--tickers', nargs='+', default=None,
        help='List of tickers to scan (default: top 10)'
    )
    parser.add_argument(
        '--train_epochs', type=int, default=5,
        help='Epochs for quick training (default: 5)'
    )
    parser.add_argument(
        '--period', type=str, default='2y',
        help='Data period (default: 2y)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for reports'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu)'
    )

    args = parser.parse_args()

    # Initialize scanner
    scanner = SaudiMarketScanner(
        tickers=args.tickers,
        device=args.device,
        train_epochs=args.train_epochs,
        data_period=args.period
    )

    # Run scan
    results = scanner.run_scan()

    # Save report
    if not results.empty:
        scanner.save_report(results, output_dir=args.output_dir)

        # Print summary table
        print("\n" + "=" * 80)
        print("SAUDI MARKET SCANNER REPORT")
        print("=" * 80)
        print(results[['Ticker', 'Name', 'Close_Price', 'Exp_Return', 'Confidence', 'AI_Signal']].to_string(index=False))
        print("=" * 80)
    else:
        logger.warning("No results to save.")

    return results


if __name__ == "__main__":
    main()
