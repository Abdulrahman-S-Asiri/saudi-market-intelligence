"""
Chronos-2 model wrapper for time series forecasting
Amazon's 120M-parameter foundation model for zero-shot forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import os

# Lazy import for optional dependency
HAS_CHRONOS = False
ChronosPipeline = None

try:
    from chronos import ChronosPipeline as _ChronosPipeline
    HAS_CHRONOS = True
    ChronosPipeline = _ChronosPipeline
except ImportError:
    pass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MODEL_SAVE_PATH


class ChronosPredictor:
    """
    Chronos-2 model wrapper for stock price forecasting

    Features:
    - Lazy model loading (loads on first prediction)
    - Probabilistic predictions with confidence intervals
    - Zero-shot forecasting (no training required)
    - CPU-only inference support
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        prediction_length: int = 5,
        context_length: int = 60,
        quantile_levels: List[float] = None,
        device: str = "cpu"
    ):
        """
        Initialize Chronos predictor

        Args:
            model_name: Hugging Face model name (amazon/chronos-t5-small, -base, -large)
            prediction_length: Number of days to forecast
            context_length: Number of historical days to use as context
            quantile_levels: Quantile levels for probabilistic forecasts
            device: Device to run inference on ('cpu' or 'cuda')
        """
        if not HAS_CHRONOS:
            raise ImportError(
                "Chronos not installed. Install with: pip install chronos-forecasting"
            )

        self.model_name = model_name
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        self.device = device

        # Lazy loading - model loaded on first prediction
        self._pipeline = None
        self._model_loaded = False

    def _load_model(self):
        """Load model lazily on first use"""
        if self._model_loaded:
            return

        print(f"Loading Chronos model: {self.model_name}...")
        print("(This may take a few minutes on first run due to model download)")

        try:
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype="float32" if self.device == "cpu" else "auto"
            )
            self._model_loaded = True
            print(f"Chronos model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Chronos model: {e}")

    def _prepare_context(self, df: pd.DataFrame, column: str = "Close") -> np.ndarray:
        """
        Prepare historical context for Chronos

        Args:
            df: DataFrame with OHLCV data
            column: Column to use for forecasting

        Returns:
            NumPy array of context values
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Get recent history (context_length days)
        context = df[column].values[-self.context_length:]

        # Handle any NaN values
        context = np.nan_to_num(context, nan=np.nanmean(context))

        return context.astype(np.float32)

    def predict(
        self,
        df: pd.DataFrame,
        column: str = "Close",
        num_samples: int = 20
    ) -> Dict:
        """
        Generate probabilistic forecast

        Args:
            df: DataFrame with historical OHLCV data
            column: Column to forecast
            num_samples: Number of forecast samples for uncertainty estimation

        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Lazy load model
        self._load_model()

        # Prepare context
        context = self._prepare_context(df, column)

        # Convert to torch tensor
        if HAS_TORCH:
            import torch
            context_tensor = torch.tensor(context).unsqueeze(0)  # Add batch dimension
        else:
            raise ImportError("PyTorch is required for Chronos")

        # Generate forecast samples
        forecast_samples = self._pipeline.predict(
            context_tensor,
            prediction_length=self.prediction_length,
            num_samples=num_samples
        )

        # Convert to numpy
        forecast_np = forecast_samples.numpy().squeeze()  # Shape: (num_samples, prediction_length)

        # Calculate quantiles
        quantiles = {}
        for q in self.quantile_levels:
            quantiles[f"q{int(q*100)}"] = np.quantile(forecast_np, q, axis=0)

        # Calculate statistics
        median = quantiles.get("q50", np.median(forecast_np, axis=0))
        mean = np.mean(forecast_np, axis=0)
        std = np.std(forecast_np, axis=0)

        # Get current price for reference
        current_price = float(df[column].iloc[-1])

        return {
            "current_price": current_price,
            "prediction_length": self.prediction_length,
            "median": median.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "quantiles": {k: v.tolist() for k, v in quantiles.items()},
            "low_10pct": float(quantiles["q10"][-1]) if "q10" in quantiles else None,
            "high_90pct": float(quantiles["q90"][-1]) if "q90" in quantiles else None,
            "predicted_price": float(median[-1]),  # Final day median prediction
        }

    def predict_direction(
        self,
        df: pd.DataFrame,
        column: str = "Close"
    ) -> Tuple[str, float, Dict]:
        """
        Predict price direction with confidence

        Simplified interface matching LSTM/Ensemble signature

        Args:
            df: DataFrame with historical data
            column: Column to forecast

        Returns:
            Tuple of (direction, confidence, details)
        """
        # Get full prediction
        prediction = self.predict(df, column)

        current_price = prediction["current_price"]
        predicted_price = prediction["predicted_price"]
        low_price = prediction.get("low_10pct", predicted_price)
        high_price = prediction.get("high_90pct", predicted_price)

        # Calculate percent change
        percent_change = ((predicted_price - current_price) / current_price) * 100

        # Determine direction
        if percent_change > 0.5:
            direction = "UP"
        elif percent_change < -0.5:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        # Calculate confidence based on prediction interval width
        # Narrower interval = higher confidence
        interval_width = (high_price - low_price) / current_price

        # Base confidence from direction strength
        direction_confidence = min(abs(percent_change) / 5, 0.5)  # Max 0.5 from direction

        # Interval confidence (narrower = more confident)
        interval_confidence = max(0, 0.5 - interval_width)  # Max 0.5 from interval

        confidence = min(direction_confidence + interval_confidence, 1.0)

        # Build details dict
        details = {
            "model": "chronos",
            "model_name": self.model_name,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "percent_change": round(percent_change, 2),
            "price_range": {
                "low_10pct": round(low_price, 2),
                "median": round(predicted_price, 2),
                "high_90pct": round(high_price, 2)
            },
            "horizon_days": self.prediction_length,
            "forecast_path": {
                "median": prediction["median"],
                "low": prediction["quantiles"].get("q10", []),
                "high": prediction["quantiles"].get("q90", [])
            }
        }

        return direction, confidence, details

    @property
    def is_available(self) -> bool:
        """Check if Chronos is available"""
        return HAS_CHRONOS

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded


def check_chronos_availability() -> Dict:
    """
    Check if Chronos is installed and available

    Returns:
        Dictionary with availability status and details
    """
    result = {
        "available": HAS_CHRONOS,
        "torch_available": HAS_TORCH,
        "message": ""
    }

    if not HAS_CHRONOS:
        result["message"] = "Chronos not installed. Install with: pip install chronos-forecasting"
    elif not HAS_TORCH:
        result["message"] = "PyTorch not installed. Install with: pip install torch"
    else:
        result["message"] = "Chronos is available and ready to use"

    return result


if __name__ == "__main__":
    print("Testing Chronos Model...")
    print("=" * 60)

    # Check availability
    status = check_chronos_availability()
    print(f"Chronos available: {status['available']}")
    print(f"Message: {status['message']}")

    if status['available']:
        # Create dummy data
        print("\nCreating test data...")
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = 30 + np.cumsum(np.random.randn(100) * 0.5)  # Random walk around 30

        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Open': prices - 0.1,
            'High': prices + 0.2,
            'Low': prices - 0.2,
            'Volume': np.random.randint(1000000, 5000000, 100)
        })

        # Create predictor
        print("\nInitializing Chronos predictor...")
        predictor = ChronosPredictor(
            model_name="amazon/chronos-t5-small",
            prediction_length=5,
            context_length=60
        )

        # Make prediction
        print("\nGenerating forecast...")
        direction, confidence, details = predictor.predict_direction(df)

        print(f"\nPrediction Results:")
        print(f"  Direction: {direction}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Current Price: {details['current_price']:.2f}")
        print(f"  Predicted Price: {details['predicted_price']:.2f}")
        print(f"  Price Range: {details['price_range']['low_10pct']:.2f} - {details['price_range']['high_90pct']:.2f}")
        print(f"  Horizon: {details['horizon_days']} days")

        print("\nChronos model test completed!")
    else:
        print("\nSkipping test - Chronos not installed")
