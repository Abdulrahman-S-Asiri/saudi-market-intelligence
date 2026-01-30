# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
FastAPI server for Saudi Stock AI Analyzer
Provides REST API endpoints for the React frontend
Uses Advanced LSTM model (BiLSTM with Multi-Head Attention)
"""

import sys
import os
import asyncio
import json
import time
import threading
from typing import Optional, List, Dict, Set
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import StockAnalyzer
from utils.config import API_CONFIG, SAUDI_STOCKS, SECTORS
from database import PositionManager, PositionCreate, PositionUpdate, PositionClose

# Import MC Dropout for calibrated confidence
import torch
import numpy as np
from models.advanced_lstm import AdvancedStockPredictor

# Import Market Regime Detector (HMM-based)
from models.market_regime import get_regime_detector, detect_market_regime

# MC Dropout Confidence Cache (to avoid re-computing for same stock)
mc_dropout_cache: Dict[str, Dict] = {}
MC_DROPOUT_CACHE_TTL = 300  # 5 minutes


def get_mc_dropout_confidence(
    symbol: str,
    data: np.ndarray,
    num_features: int,
    n_samples: int = 10
) -> Dict:
    """
    Run Monte Carlo Dropout inference to get calibrated confidence.

    Uses N=10 forward passes with dropout enabled to estimate:
    - Prediction mean (final predicted return)
    - Prediction variance (model uncertainty)
    - Calibrated confidence score (0-100%)

    Args:
        symbol: Stock symbol for caching
        data: Preprocessed sequence data (seq_len, num_features)
        num_features: Number of input features
        n_samples: Number of MC samples (default: 10)

    Returns:
        Dict with prediction, variance, and calibrated confidence
    """
    import time

    # Check cache
    cache_key = f"{symbol}_{hash(data.tobytes())}"
    if cache_key in mc_dropout_cache:
        cached = mc_dropout_cache[cache_key]
        if time.time() - cached['timestamp'] < MC_DROPOUT_CACHE_TTL:
            return cached['result']

    try:
        # Create predictor instance
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = AdvancedStockPredictor(
            num_features=num_features,
            hidden_sizes=[64, 32],
            dropout=0.5,
            device=device
        )

        # Run MC Dropout prediction
        result = predictor.predict_with_calibrated_confidence(data, n_samples=n_samples)

        # Extract values
        mc_result = {
            'prediction': float(result['prediction'][0]),
            'prediction_std': float(result['prediction_std'][0]),
            'confidence': float(result['confidence'][0]),
            'variance': float(result['variance'][0])
        }

        # Cache result
        mc_dropout_cache[cache_key] = {
            'result': mc_result,
            'timestamp': time.time()
        }

        return mc_result

    except Exception as e:
        # Return default on error
        return {
            'prediction': 0.0,
            'prediction_std': 0.1,
            'confidence': 50.0,
            'variance': 0.01,
            'error': str(e)
        }


# Initialize FastAPI app
app = FastAPI(
    title="Saudi Stock AI Analyzer API",
    description="AI-powered stock analysis for Saudi Arabian market (Tadawul) - Advanced LSTM Edition",
    version="3.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer (singleton) - Now uses Advanced LSTM only
analyzer = StockAnalyzer()

# Initialize Position Manager
position_manager = PositionManager()


# ==================== Market Rankings Cache ====================

class MarketRankingsCache:
    """
    Cache for market rankings to avoid expensive full-market scans.
    Rankings are updated once per hour or on-demand.
    """

    def __init__(self, cache_duration_seconds: int = 3600):
        self.cache_duration = cache_duration_seconds  # Default: 1 hour
        self.last_update: Optional[datetime] = None
        self.cached_rankings: Optional[Dict] = None
        self._lock = threading.Lock()
        self._is_updating = False

    def is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if self.last_update is None or self.cached_rankings is None:
            return False
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return elapsed < self.cache_duration

    def get_cached_rankings(self) -> Optional[Dict]:
        """Get cached rankings if valid"""
        if self.is_cache_valid():
            return self.cached_rankings
        return None

    def update_rankings(self, rankings: Dict):
        """Update the cache with new rankings"""
        with self._lock:
            self.cached_rankings = rankings
            self.last_update = datetime.now()
            self._is_updating = False

    def get_cache_age_seconds(self) -> Optional[float]:
        """Get age of cache in seconds"""
        if self.last_update is None:
            return None
        return (datetime.now() - self.last_update).total_seconds()

    def set_updating(self, status: bool):
        """Set updating flag"""
        self._is_updating = status

    def is_updating(self) -> bool:
        """Check if rankings are being updated"""
        return self._is_updating


# Initialize market rankings cache (1 hour TTL)
market_rankings_cache = MarketRankingsCache(cache_duration_seconds=3600)

# Top stocks to scan for quick rankings (most liquid/important stocks)
TOP_STOCKS_FOR_RANKING = [
    # Energy
    "2222",  # Saudi Aramco
    # Banks
    "1120", "1180", "1010", "1140", "1182",  # Al Rajhi, Alinma, Riyad, Albilad, SNB
    # Materials
    "2010", "2350", "2310", "1211",  # SABIC, Saudi Kayan, Sipchem, Ma'aden
    # Telecom
    "7010", "7020", "7030",  # STC, Mobily, Zain
    # Utilities
    "5110", "2082",  # SEC, ACWA Power
    # Consumer
    "2280", "4190", "4001",  # Almarai, Jarir, Al Othaim
    # Healthcare
    "4013", "4002",  # Dr. Sulaiman Al Habib, Mouwasat
    # Real Estate
    "4300", "4250",  # Dar Al Arkan, Jabal Omar
    # Insurance
    "8210", "8010",  # Bupa, Tawuniya
    # REITs
    "4330", "4331",  # Riyad REIT, Jadwa REIT
    # Food
    "2050", "6002",  # Savola, Herfy
    # Capital Goods
    "2320", "2240",  # Al Babtain, Zamil
    # Transportation
    "4030", "4261",  # Bahri, Theeb
    # Software
    "7201", "7200",  # Elm, Solutions by STC
]


def scan_stocks_for_rankings(
    stock_symbols: List[str],
    analyzer_instance,
    use_ml_confidence: bool = True
) -> Dict:
    """
    Scan a list of stocks and return rankings based on technical indicators
    with Monte Carlo Dropout confidence estimation.

    NO STRICT THRESHOLDS - Returns top 5 gainers and losers regardless of confidence.
    Ranks ALL stocks by predicted_change and returns the best available options.

    Args:
        stock_symbols: List of stock symbols to scan
        analyzer_instance: StockAnalyzer instance
        use_ml_confidence: If True, uses Monte Carlo Dropout for scientific confidence.
                          If False, uses heuristic confidence (faster but less accurate).

    Returns:
        Dict with top_bullish, top_bearish, and metadata
    """
    results = []
    errors = []

    for symbol in stock_symbols:
        try:
            # Use train_model=True when ML confidence is needed, False for speed
            analysis = analyzer_instance.analyze_stock(
                symbol,
                period="3mo",
                train_model=use_ml_confidence,
                force_retrain=False,
                include_macro=False
            )

            if 'error' in analysis:
                errors.append({"symbol": symbol, "error": analysis['error']})
                continue

            signal = analysis.get('signal', {})
            indicators = analysis.get('indicators', {})
            performance = analysis.get('performance', {})
            ml_prediction = analysis.get('ml_prediction', {})

            # Get key indicators
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            signal_action = signal.get('action', 'HOLD')
            signal_confidence = signal.get('confidence', 50)
            volatility = performance.get('volatility', 15)
            total_return = performance.get('total_return', 0)

            # Calculate predicted change based on multiple factors
            # This creates a continuous score (not binary direction)

            # RSI component: -1 (overbought) to +1 (oversold)
            # RSI 30 = +1, RSI 50 = 0, RSI 70 = -1
            rsi_component = (50 - rsi) / 40  # Normalized: oversold=positive, overbought=negative
            rsi_component = max(-1, min(1, rsi_component))  # Clamp to [-1, 1]

            # MACD component: positive MACD = bullish, negative = bearish
            macd_component = 0
            if macd != 0:
                macd_component = 0.5 if macd > macd_signal else -0.5

            # Recent momentum from total return
            momentum_component = max(-0.5, min(0.5, total_return / 20))  # Normalize recent return

            # Combined score (weighted average)
            combined_score = (rsi_component * 0.5) + (macd_component * 0.3) + (momentum_component * 0.2)

            # Predicted percentage change based on score and volatility
            # Higher volatility = larger potential move
            volatility_factor = min(volatility / 100 * 2, 0.06)  # Cap at 6%
            predicted_change = combined_score * volatility_factor * 100

            # ============================================================
            # MONTE CARLO DROPOUT CONFIDENCE (Scientific Approach)
            # ============================================================
            # Instead of heuristic confidence, use ML model uncertainty
            #
            # When use_ml_confidence=True:
            #   1. Get ML prediction variance from trained model
            #   2. Map variance to calibrated 0-100% confidence
            #   3. Low variance = High confidence, High variance = Low confidence
            #
            # Formula: confidence = 100 * exp(-k * variance)
            # ============================================================

            if use_ml_confidence and ml_prediction:
                # Use ML prediction metrics for confidence
                ml_conf = ml_prediction.get('confidence', 50)
                ml_variance = ml_prediction.get('uncertainty', 0.05)

                # If we have actual uncertainty from the model, use calibrated formula
                if ml_variance and ml_variance > 0:
                    # Calibration constant (tuned for stock returns)
                    # k=10 means: std=0.1 → 37% conf, std=0.05 → 78% conf
                    k = 10.0
                    confidence_raw = 100.0 * np.exp(-k * ml_variance)
                    confidence = float(np.clip(confidence_raw, 20, 95))
                else:
                    # Use model's reported confidence if no variance
                    confidence = float(np.clip(ml_conf * 100, 20, 95))

                # Also use ML predicted direction to adjust predicted_change
                ml_direction = ml_prediction.get('direction', 0)
                ml_pred_value = ml_prediction.get('prediction', 0)
                if ml_pred_value != 0:
                    # Blend technical and ML predictions (60% ML, 40% technical)
                    predicted_change = 0.6 * (ml_pred_value * 100) + 0.4 * predicted_change
            else:
                # Fallback: Heuristic confidence based on indicator agreement
                # This is less accurate but much faster
                indicator_strength = abs(rsi_component) * 50 + abs(macd_component) * 30
                confidence = min(max(indicator_strength + 20, 20), 95)

            # Determine direction label
            if predicted_change > 0.5:
                direction = 'UP'
            elif predicted_change < -0.5:
                direction = 'DOWN'
            else:
                direction = 'NEUTRAL'

            result = {
                "symbol": symbol,
                "name": analysis.get('name', symbol),
                "sector": analysis.get('sector', 'Unknown'),
                "price": signal.get('price', 0),
                "direction": direction,
                "ml_confidence": round(confidence, 1),
                "predicted_change": round(predicted_change, 2),
                "signal_action": signal_action,
                "signal_confidence": signal_confidence,
                "market_regime": signal.get('market_regime', 'Unknown'),
                "rsi": round(rsi, 1),
                "total_return": round(total_return, 2),
                "volatility": round(volatility, 2),
                "confidence_method": "mc_dropout" if use_ml_confidence else "heuristic"
            }

            results.append(result)

        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
            continue

    # IMPORTANT: No filtering by threshold!
    # Sort ALL stocks by predicted_change and take top/bottom 5

    # Top Gainers: Sort by predicted_change DESCENDING (highest first)
    sorted_by_gain = sorted(results, key=lambda x: x['predicted_change'], reverse=True)
    top_bullish = sorted_by_gain[:5]

    # Top Losers: Sort by predicted_change ASCENDING (most negative first)
    sorted_by_loss = sorted(results, key=lambda x: x['predicted_change'])
    top_bearish = sorted_by_loss[:5]

    # Update direction labels for display consistency
    for stock in top_bullish:
        if stock['predicted_change'] > 0:
            stock['direction'] = 'UP'
    for stock in top_bearish:
        if stock['predicted_change'] < 0:
            stock['direction'] = 'DOWN'

    return {
        "top_bullish": top_bullish,
        "top_bearish": top_bearish,
        "total_scanned": len(results),
        "total_errors": len(errors),
        "errors": errors[:5] if errors else [],
        "timestamp": datetime.now().isoformat(),
        "stocks_analyzed": [r['symbol'] for r in results],
        "confidence_method": "mc_dropout" if use_ml_confidence else "heuristic"
    }


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Saudi Stock AI Analyzer API",
        "version": "3.0.0",
        "model": "Advanced LSTM (BiLSTM + Multi-Head Attention)",
        "endpoints": {
            "stocks": "/api/stocks",
            "sectors": "/api/sectors",
            "stocks_by_sector": "/api/stocks/sector/{sector}",
            "analyze": "/api/analyze/{symbol}",
            "chart": "/api/chart/{symbol}",
            "predict": "/api/predict/{symbol}",
            "backtest": "/api/backtest/{symbol}",
            "signals_history": "/api/signals/history/{symbol}",
            "compare": "/api/compare",
            "risk": "/api/risk/{symbol}",
            "health": "/api/health",
            "models": "/api/models",
            "market_rankings": {
                "get": "/api/market-rankings",
                "status": "/api/market-rankings/status",
                "refresh": "/api/market-rankings/refresh (POST)"
            },
            "positions": {
                "list": "/api/positions",
                "create": "/api/positions (POST)",
                "get": "/api/positions/{id}",
                "update": "/api/positions/{id} (PUT)",
                "close": "/api/positions/{id}/close (PUT)",
                "delete": "/api/positions/{id} (DELETE)",
                "from_signal": "/api/positions/from-signal/{symbol}",
                "summary": "/api/positions/summary"
            }
        }
    }


@app.get("/api/stocks")
async def get_stocks():
    """Get list of available Saudi stocks"""
    stocks = analyzer.get_available_stocks()
    return {
        "count": len(stocks),
        "stocks": stocks
    }


@app.get("/api/sectors")
async def get_sectors():
    """Get all available sectors with their stocks"""
    return {
        "count": len(SECTORS),
        "sectors": SECTORS
    }


@app.get("/api/stocks/sector/{sector}")
async def get_stocks_by_sector(sector: str):
    """Get stocks filtered by sector"""
    # Find matching sector (case-insensitive)
    matching_sector = None
    for s in SECTORS.keys():
        if s.lower() == sector.lower() or s.lower().replace(" ", "-") == sector.lower():
            matching_sector = s
            break

    if not matching_sector:
        raise HTTPException(status_code=404, detail=f"Sector '{sector}' not found")

    sector_stocks = SECTORS[matching_sector]
    stocks_info = {
        symbol: SAUDI_STOCKS.get(symbol, {"name": "Unknown", "sector": matching_sector})
        for symbol in sector_stocks
        if symbol in SAUDI_STOCKS
    }

    return {
        "sector": matching_sector,
        "count": len(stocks_info),
        "stocks": stocks_info
    }


@app.get("/api/analyze/{symbol}")
async def analyze_stock(
    symbol: str,
    period: str = Query(default="6mo", description="Data period (1mo, 3mo, 6mo, 1y, 2y)"),
    train_model: bool = Query(default=True, description="Whether to train ML model"),
    force_retrain: bool = Query(default=False, description="Force model retraining")
):
    """
    Perform complete analysis on a stock using Advanced LSTM model

    - **symbol**: Stock symbol (e.g., 2222 for Aramco)
    - **period**: Historical data period
    - **train_model**: Whether to include ML prediction
    - **force_retrain**: Force model retraining even if cached
    """
    try:
        result = analyzer.analyze_stock(
            symbol,
            period=period,
            train_model=train_model,
            force_retrain=force_retrain
        )

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chart/{symbol}")
async def get_chart_data(
    symbol: str,
    period: str = Query(default="6mo", description="Data period")
):
    """
    Get chart data for visualization

    Returns OHLCV data with technical indicators
    """
    try:
        result = analyzer.get_chart_data(symbol, period=period)

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/{symbol}")
async def predict_stock(
    symbol: str,
    period: str = Query(default="1y", description="Training data period")
):
    """
    Get ML prediction for a stock using Advanced LSTM

    Note: This endpoint trains the model if not cached
    """
    try:
        result = analyzer.analyze_stock(symbol, period=period, train_model=True)

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        return {
            "symbol": result['symbol'],
            "name": result['name'],
            "current_price": result['signal']['price'],
            "signal": result['signal'],
            "ml_prediction": result.get('ml_prediction'),
            "trend": result['trend'],
            "indicators": result['indicators']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/{symbol}")
async def backtest_stock(
    symbol: str,
    period: str = Query(default="1y", description="Backtest period (6mo, 1y, 2y)"),
    min_confidence: float = Query(default=60, description="Minimum confidence to take trades (0-100)"),
    hold_period: int = Query(default=5, description="Default holding period in days")
):
    """
    Run backtest on historical data

    - **symbol**: Stock symbol
    - **period**: Historical data period for backtesting
    - **min_confidence**: Minimum signal confidence to execute trades
    - **hold_period**: Default number of days to hold a position

    Returns:
    - Total trades, win rate, profit factor
    - Total return, max drawdown
    - Sharpe and Sortino ratios
    - Signal accuracy metrics
    """
    try:
        result = analyzer.run_backtest(
            symbol,
            period=period,
            min_confidence=min_confidence,
            hold_period=hold_period
        )

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals/history/{symbol}")
async def get_signal_history(
    symbol: str,
    limit: int = Query(default=20, description="Number of signals to return (max 100)")
):
    """
    Get historical signals for a stock

    - **symbol**: Stock symbol
    - **limit**: Maximum number of signals to return

    Returns list of past signals with:
    - Timestamp, signal type, confidence
    - Price at signal time
    - Indicator values
    - Outcome (if evaluated)
    """
    try:
        # Limit to 100 max
        limit = min(limit, 100)

        signals = analyzer.get_signal_history(symbol, limit=limit)

        return {
            "symbol": symbol,
            "count": len(signals),
            "signals": signals
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compare")
async def compare_stocks(
    symbols: str = Query(description="Comma-separated list of stock symbols (e.g., 2222,1120,2010)"),
    period: str = Query(default="6mo", description="Data period for comparison")
):
    """
    Compare multiple stocks

    - **symbols**: Comma-separated list of stock symbols
    - **period**: Historical data period

    Returns comparison of:
    - Current signal and confidence
    - Price and returns
    - Volatility and Sharpe ratio
    - RSI values
    """
    try:
        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

        if len(symbol_list) < 2:
            raise HTTPException(
                status_code=400,
                detail="Please provide at least 2 symbols for comparison"
            )

        if len(symbol_list) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 symbols allowed for comparison"
            )

        result = analyzer.compare_stocks(symbol_list, period=period)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/{symbol}")
async def get_risk_metrics(
    symbol: str,
    period: str = Query(default="1y", description="Data period for risk calculation")
):
    """
    Get comprehensive risk metrics for a stock

    - **symbol**: Stock symbol
    - **period**: Historical data period

    Returns:
    - Volatility, max drawdown, drawdown duration
    - Sharpe, Sortino, Calmar ratios
    - VaR (95% and 99%)
    - Expected Shortfall
    - Distribution metrics (skewness, kurtosis)
    """
    try:
        result = analyzer.analyze_stock(symbol, period=period, train_model=False)

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        return {
            "symbol": symbol,
            "name": result['name'],
            "period": period,
            "risk_metrics": result['risk'],
            "performance": result['performance']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cache/clear")
async def clear_cache(
    symbol: Optional[str] = Query(default=None, description="Symbol to clear (or all if not specified)")
):
    """
    Clear model cache

    - **symbol**: Optional specific symbol to clear, or clears all if not provided
    """
    try:
        analyzer.clear_model_cache(symbol)
        return {
            "status": "success",
            "message": f"Cache cleared for {'all symbols' if symbol is None else symbol}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "saudi-stock-ai-analyzer",
        "version": "3.0.0",
        "model": {
            "type": "advanced_lstm",
            "name": "BiLSTM + Multi-Head Attention",
            "description": "Bidirectional LSTM with Multi-Head Attention, Residual Connections, and Uncertainty Estimation"
        },
        "features": {
            "model_caching": True,
            "backtesting": True,
            "risk_metrics": True,
            "monte_carlo_backtest": True,
            "uncertainty_estimation": True,
            "position_manager": True,
            "market_rankings": True
        }
    }


# ==================== Market Rankings Endpoints ====================

@app.get("/api/market-rankings")
async def get_market_rankings(
    background_tasks: BackgroundTasks,
    force_refresh: bool = Query(default=False, description="Force refresh rankings (bypass cache)"),
    quick_scan: bool = Query(default=True, description="Scan only top 30 stocks (faster)"),
    use_ml: bool = Query(default=True, description="Use Monte Carlo Dropout for scientific confidence scores")
):
    """
    Get market rankings - Top bullish and bearish stocks.

    This endpoint scans tracked stocks and ranks them by predicted movement.
    Uses Monte Carlo Dropout for scientifically calibrated confidence scores.

    - **force_refresh**: Bypass cache and scan stocks (slow)
    - **quick_scan**: Only scan top 30 most liquid stocks (faster, default: True)
    - **use_ml**: Use Monte Carlo Dropout confidence (default: True)
                  When True: Confidence = 100 * exp(-10 * variance), calibrated from model uncertainty
                  When False: Confidence = heuristic based on indicator strength (faster)

    Returns:
    - **top_bullish**: Top 5 stocks predicted to rise (sorted by predicted_change)
    - **top_bearish**: Top 5 stocks predicted to fall (sorted by predicted_change)
    - **ml_confidence**: Calibrated 0-100% confidence from MC Dropout variance
    - **cache_info**: Cache status and age

    Note: Rankings are cached for 1 hour to improve performance.
    """
    try:
        # Check cache first
        if not force_refresh:
            cached = market_rankings_cache.get_cached_rankings()
            if cached:
                cache_age = market_rankings_cache.get_cache_age_seconds()
                return {
                    **cached,
                    "cache_info": {
                        "from_cache": True,
                        "cache_age_seconds": round(cache_age, 0) if cache_age else None,
                        "cache_age_minutes": round(cache_age / 60, 1) if cache_age else None,
                        "next_refresh_minutes": round((3600 - cache_age) / 60, 1) if cache_age else None
                    }
                }

        # Check if already updating
        if market_rankings_cache.is_updating():
            # Return partial data if available, otherwise return updating status
            partial = market_rankings_cache.cached_rankings
            if partial:
                cache_age = market_rankings_cache.get_cache_age_seconds()
                return {
                    **partial,
                    "cache_info": {
                        "from_cache": True,
                        "is_updating": True,
                        "cache_age_seconds": round(cache_age, 0) if cache_age else None,
                        "cache_age_minutes": round(cache_age / 60, 1) if cache_age else None
                    }
                }
            return {
                "status": "updating",
                "message": "Rankings are currently being updated. Please try again shortly.",
                "top_bullish": [],
                "top_bearish": [],
                "total_scanned": 0,
                "cache_info": {
                    "from_cache": False,
                    "is_updating": True
                }
            }

        # Mark as updating
        market_rankings_cache.set_updating(True)

        # Determine which stocks to scan (limit for performance)
        if quick_scan:
            stocks_to_scan = TOP_STOCKS_FOR_RANKING[:15]  # Reduced to 15 for faster response
        else:
            stocks_to_scan = TOP_STOCKS_FOR_RANKING

        # Scan stocks and generate rankings
        print(f"Scanning {len(stocks_to_scan)} stocks for market rankings (ML confidence: {use_ml})...")
        rankings = scan_stocks_for_rankings(stocks_to_scan, analyzer, use_ml_confidence=use_ml)

        # Update cache
        market_rankings_cache.update_rankings(rankings)

        return {
            **rankings,
            "cache_info": {
                "from_cache": False,
                "cache_age_seconds": 0,
                "cache_age_minutes": 0,
                "next_refresh_minutes": 60,
                "scan_type": "quick" if quick_scan else "full"
            }
        }

    except Exception as e:
        market_rankings_cache.set_updating(False)
        print(f"Error in market-rankings: {str(e)}")
        # Return empty rankings instead of crashing
        return {
            "top_bullish": [],
            "top_bearish": [],
            "total_scanned": 0,
            "total_errors": 1,
            "errors": [{"error": str(e)}],
            "timestamp": datetime.now().isoformat(),
            "cache_info": {
                "from_cache": False,
                "error": str(e)
            }
        }


@app.get("/api/market-rankings/status")
async def get_rankings_status():
    """
    Get the status of market rankings cache.

    Returns cache status, age, and whether an update is in progress.
    """
    cache_age = market_rankings_cache.get_cache_age_seconds()

    return {
        "cache_valid": market_rankings_cache.is_cache_valid(),
        "is_updating": market_rankings_cache.is_updating(),
        "last_update": market_rankings_cache.last_update.isoformat() if market_rankings_cache.last_update else None,
        "cache_age_seconds": round(cache_age, 0) if cache_age else None,
        "cache_age_minutes": round(cache_age / 60, 1) if cache_age else None,
        "cache_duration_hours": market_rankings_cache.cache_duration / 3600,
        "stocks_in_quick_scan": len(TOP_STOCKS_FOR_RANKING)
    }


# ==================== Market Regime Detection (HMM) ====================

# Cache for market regime (refreshes every 15 minutes)
_regime_cache: Dict = {}
_regime_cache_time: Optional[datetime] = None
REGIME_CACHE_TTL = 900  # 15 minutes


@app.get("/api/market-status")
async def get_market_status():
    """
    Get current market regime using Hidden Markov Model (HMM) detection.

    The regime detector analyzes the TASI Index to classify the market into:
    - **Bull Market** 🟢: High positive returns, low volatility. Good for buying.
    - **Bear Market** 🔴: Negative returns, high volatility. Consider reducing exposure.
    - **Sideways** 🟡: Low returns, moderate volatility. Selective trading.

    Returns:
    - **regime**: Current regime label (bull, bear, sideways)
    - **regime_name**: Human-readable name (e.g., "Bull Market")
    - **emoji**: Visual indicator (🟢, 🔴, 🟡)
    - **confidence**: Probability of current state (0-100%)
    - **warning**: Warning message if bearish
    - **regime_probabilities**: Probabilities for all states

    This helps users decide IF they should be in the market at all.
    """
    global _regime_cache, _regime_cache_time

    # Check cache
    if _regime_cache and _regime_cache_time:
        cache_age = (datetime.now() - _regime_cache_time).total_seconds()
        if cache_age < REGIME_CACHE_TTL:
            return {
                **_regime_cache,
                "cache_info": {
                    "from_cache": True,
                    "cache_age_seconds": round(cache_age, 0)
                }
            }

    try:
        import yfinance as yf

        # Fetch TASI Index data (1 year for training)
        print("Fetching TASI Index for regime detection...")
        tasi_data = yf.download("^TASI.SR", period="1y", progress=False)

        if tasi_data.empty:
            return {
                "regime": "sideways",
                "regime_name": "Sideways",
                "emoji": "🟡",
                "color": "#ffc107",
                "confidence": 50.0,
                "warning": None,
                "regime_probabilities": {"bull": 33.3, "bear": 33.3, "sideways": 33.4},
                "error": "Could not fetch TASI data",
                "cache_info": {"from_cache": False}
            }

        # Detect regime using HMM
        regime_result = detect_market_regime(tasi_data)

        # Add TASI summary stats
        current_price = float(tasi_data['Close'].iloc[-1])
        prev_close = float(tasi_data['Close'].iloc[-2])
        daily_change = (current_price - prev_close) / prev_close * 100

        # Weekly and monthly returns
        week_ago_price = float(tasi_data['Close'].iloc[-5]) if len(tasi_data) >= 5 else current_price
        month_ago_price = float(tasi_data['Close'].iloc[-22]) if len(tasi_data) >= 22 else current_price
        weekly_return = (current_price - week_ago_price) / week_ago_price * 100
        monthly_return = (current_price - month_ago_price) / month_ago_price * 100

        # Build response
        result = {
            **regime_result,
            "tasi": {
                "price": round(current_price, 2),
                "daily_change": round(daily_change, 2),
                "weekly_return": round(weekly_return, 2),
                "monthly_return": round(monthly_return, 2)
            },
            "cache_info": {"from_cache": False}
        }

        # Update cache
        _regime_cache = result
        _regime_cache_time = datetime.now()

        return result

    except Exception as e:
        print(f"Error in market-status: {e}")
        import traceback
        traceback.print_exc()

        return {
            "regime": "sideways",
            "regime_name": "Sideways",
            "emoji": "🟡",
            "color": "#ffc107",
            "confidence": 50.0,
            "warning": None,
            "error": str(e),
            "cache_info": {"from_cache": False}
        }


@app.post("/api/market-status/refresh")
async def refresh_market_status():
    """
    Force refresh the market regime detection.

    Clears the cache and re-runs the HMM regime detection.
    """
    global _regime_cache, _regime_cache_time

    # Clear cache
    _regime_cache = {}
    _regime_cache_time = None

    # Call get_market_status to refresh
    return await get_market_status()


@app.post("/api/market-rankings/refresh")
async def refresh_market_rankings(
    background_tasks: BackgroundTasks,
    quick_scan: bool = Query(default=True, description="Scan only top stocks"),
    use_ml: bool = Query(default=True, description="Use Monte Carlo Dropout confidence")
):
    """
    Trigger a background refresh of market rankings.

    Use this to manually trigger a rankings update without waiting.

    - **quick_scan**: Only scan top stocks (faster)
    - **use_ml**: Use Monte Carlo Dropout for scientific confidence scores
    """
    if market_rankings_cache.is_updating():
        return {
            "status": "already_updating",
            "message": "Rankings update already in progress"
        }

    def background_update():
        """Background task to update rankings"""
        try:
            market_rankings_cache.set_updating(True)
            stocks_to_scan = TOP_STOCKS_FOR_RANKING if quick_scan else list(SAUDI_STOCKS.keys())[:50]
            rankings = scan_stocks_for_rankings(stocks_to_scan, analyzer, use_ml_confidence=use_ml)
            market_rankings_cache.update_rankings(rankings)
        except Exception as e:
            print(f"Background rankings update failed: {e}")
            market_rankings_cache.set_updating(False)

    background_tasks.add_task(background_update)

    return {
        "status": "started",
        "message": f"Rankings refresh started for {len(TOP_STOCKS_FOR_RANKING if quick_scan else [])} stocks",
        "scan_type": "quick" if quick_scan else "full",
        "confidence_method": "mc_dropout" if use_ml else "heuristic"
    }


# ==================== Position Manager Endpoints ====================

@app.get("/api/positions")
async def get_positions(
    status: Optional[str] = Query(default=None, description="Filter by status: OPEN or CLOSED")
):
    """
    Get all positions with live P&L

    - **status**: Optional filter by position status (OPEN/CLOSED)

    Returns list of positions with:
    - Entry price, current price, exit price
    - P&L amount and percentage
    - WIN/LOSS/HOLD status
    """
    try:
        positions = position_manager.get_all_positions(status=status)
        summary = position_manager.get_summary()

        return {
            "positions": positions,
            "summary": summary,
            "count": len(positions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/positions")
async def create_position(position: PositionCreate):
    """
    Create a new position manually

    - **symbol**: Stock symbol (e.g., 2222)
    - **entry_price**: Entry price per share
    - **quantity**: Number of shares (default: 1.0)
    - **notes**: Optional notes
    """
    try:
        new_position = position_manager.create_position(position)
        return {
            "success": True,
            "position": new_position,
            "message": f"Position created for {position.symbol}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions/{position_id}")
async def get_position(position_id: int):
    """Get a single position by ID"""
    try:
        position = position_manager.get_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
        return position
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/positions/{position_id}")
async def update_position(position_id: int, update_data: PositionUpdate):
    """
    Update a position

    - **current_price**: Update current price (recalculates P&L)
    - **quantity**: Update quantity
    - **notes**: Update notes
    """
    try:
        position = position_manager.update_position(position_id, update_data)
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
        return {
            "success": True,
            "position": position
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/positions/{position_id}/close")
async def close_position(position_id: int, close_data: PositionClose):
    """
    Close a position (mark as WIN/LOSS)

    - **exit_price**: The price at which the position was closed
    """
    try:
        position = position_manager.close_position(position_id, close_data)
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
        return {
            "success": True,
            "position": position,
            "message": f"Position closed with result: {position['result']}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/positions/{position_id}")
async def delete_position(position_id: int):
    """Delete a position"""
    try:
        deleted = position_manager.delete_position(position_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
        return {
            "success": True,
            "message": f"Position {position_id} deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/positions/from-signal/{symbol}")
async def create_position_from_signal(
    symbol: str,
    price: float = Query(..., description="Entry price from signal"),
    signal_id: Optional[str] = Query(default=None, description="Signal identifier")
):
    """
    Create a position from a BUY signal

    - **symbol**: Stock symbol
    - **price**: Entry price from the signal
    - **signal_id**: Optional signal identifier for tracking
    """
    try:
        position = position_manager.create_from_signal(symbol, price, signal_id)
        return {
            "success": True,
            "position": position,
            "message": f"Position created from signal for {symbol}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions/summary")
async def get_positions_summary():
    """Get summary statistics for all positions"""
    try:
        summary = position_manager.get_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/positions/update-prices/{symbol}")
async def update_position_prices(
    symbol: str,
    current_price: float = Query(..., description="Current market price")
):
    """
    Update current price for all open positions of a symbol

    - **symbol**: Stock symbol
    - **current_price**: Current market price
    """
    try:
        updated_count = position_manager.update_prices_for_symbol(symbol, current_price)
        return {
            "success": True,
            "updated_count": updated_count,
            "message": f"Updated {updated_count} positions for {symbol}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Model Endpoints ====================

@app.get("/api/models")
async def get_available_models():
    """
    List available prediction models with status

    Returns only Advanced LSTM model info (other models have been removed)
    """
    models = analyzer.get_available_models()
    current = analyzer.current_model_type

    return {
        "current_model": current,
        "models": models
    }


@app.post("/api/models/select")
async def select_model(
    model_type: str = Query(description="Model type (only 'advanced_lstm' is supported)")
):
    """
    Model selection endpoint (deprecated)

    This endpoint is kept for backwards compatibility but only supports advanced_lstm.
    All other model types have been removed from the system.
    """
    if model_type != 'advanced_lstm':
        return {
            "success": False,
            "message": f"Model '{model_type}' is not available. Only 'advanced_lstm' is supported.",
            "available_models": ["advanced_lstm"]
        }

    return {
        "success": True,
        "model_type": "advanced_lstm",
        "message": "Using Advanced LSTM (BiLSTM + Multi-Head Attention) model"
    }


# ==================== WebSocket Manager ====================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # symbol -> set of connections
        self.connection_symbols: Dict[WebSocket, Set[str]] = {}  # connection -> set of symbols

    async def connect(self, websocket: WebSocket, symbol: str = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        if websocket not in self.connection_symbols:
            self.connection_symbols[websocket] = set()

        if symbol:
            await self.subscribe(websocket, symbol)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.connection_symbols:
            for symbol in self.connection_symbols[websocket]:
                if symbol in self.active_connections:
                    self.active_connections[symbol].discard(websocket)
            del self.connection_symbols[websocket]

    async def subscribe(self, websocket: WebSocket, symbol: str):
        """Subscribe a connection to a symbol."""
        if symbol not in self.active_connections:
            self.active_connections[symbol] = set()
        self.active_connections[symbol].add(websocket)
        self.connection_symbols[websocket].add(symbol)

    async def unsubscribe(self, websocket: WebSocket, symbol: str):
        """Unsubscribe a connection from a symbol."""
        if symbol in self.active_connections:
            self.active_connections[symbol].discard(websocket)
        if websocket in self.connection_symbols:
            self.connection_symbols[websocket].discard(symbol)

    async def broadcast_to_symbol(self, symbol: str, message: dict):
        """Broadcast a message to all connections subscribed to a symbol."""
        if symbol in self.active_connections:
            disconnected = []
            for connection in self.active_connections[symbol]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)

            for conn in disconnected:
                self.disconnect(conn)

    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)


# Initialize WebSocket manager
ws_manager = ConnectionManager()


# ==================== WebSocket Endpoints ====================

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time stock updates.

    Sends:
    - price_update: Real-time price changes
    - analysis_update: Full analysis updates
    - signal_alert: Trading signal alerts
    """
    await ws_manager.connect(websocket, symbol)

    try:
        # Send initial data
        try:
            result = analyzer.analyze_stock(symbol, period="1mo", train_model=False)
            await ws_manager.send_personal_message(websocket, {
                "type": "analysis_update",
                "data": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            await ws_manager.send_personal_message(websocket, {
                "type": "error",
                "message": str(e)
            })

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "subscribe":
                new_symbol = message.get("symbol")
                if new_symbol:
                    await ws_manager.subscribe(websocket, new_symbol)
                    # Send initial data for new symbol
                    try:
                        result = analyzer.analyze_stock(new_symbol, period="1mo", train_model=False)
                        await ws_manager.send_personal_message(websocket, {
                            "type": "analysis_update",
                            "data": result,
                            "symbol": new_symbol,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception:
                        pass

            elif message.get("type") == "unsubscribe":
                old_symbol = message.get("symbol")
                if old_symbol:
                    await ws_manager.unsubscribe(websocket, old_symbol)

            elif message.get("type") == "refresh":
                # Send fresh analysis
                for sym in ws_manager.connection_symbols.get(websocket, []):
                    try:
                        result = analyzer.analyze_stock(sym, period="1mo", train_model=False)
                        await ws_manager.send_personal_message(websocket, {
                            "type": "analysis_update",
                            "data": result,
                            "symbol": sym,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception:
                        pass

            elif message.get("type") == "ping":
                await ws_manager.send_personal_message(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/multi")
async def websocket_multi_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for multi-symbol subscriptions.

    Allows subscribing to multiple symbols at once.
    """
    await ws_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "subscribe_multi":
                symbols = message.get("symbols", [])
                for symbol in symbols:
                    await ws_manager.subscribe(websocket, symbol)
                    try:
                        result = analyzer.analyze_stock(symbol, period="1mo", train_model=False)
                        await ws_manager.send_personal_message(websocket, {
                            "type": "analysis_update",
                            "data": result,
                            "symbol": symbol,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception:
                        pass

            elif message.get("type") == "ping":
                await ws_manager.send_personal_message(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# Background task for periodic updates (optional)
async def periodic_price_updates():
    """Background task to send periodic price updates to connected clients."""
    while True:
        await asyncio.sleep(30)  # Update every 30 seconds

        for symbol, connections in ws_manager.active_connections.items():
            if connections:
                try:
                    # Get latest price
                    result = analyzer.analyze_stock(symbol, period="1mo", train_model=False)
                    if result and 'signal' in result:
                        message = {
                            "type": "price_update",
                            "symbol": symbol,
                            "price": result['signal'].get('price'),
                            "timestamp": datetime.now().isoformat()
                        }
                        await ws_manager.broadcast_to_symbol(symbol, message)
                except Exception:
                    pass


# Start background task on startup
@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    # Uncomment to enable periodic updates
    # asyncio.create_task(periodic_price_updates())
    pass


# ==================== Advanced Backtest Endpoint ====================

@app.get("/api/backtest/advanced/{symbol}")
async def advanced_backtest(
    symbol: str,
    period: str = Query(default="2y", description="Backtest period"),
    n_simulations: int = Query(default=100, description="Monte Carlo simulations")
):
    """
    Run advanced backtest with Monte Carlo simulation.

    Returns probabilistic metrics and confidence intervals.
    """
    try:
        result = analyzer.run_advanced_backtest(
            symbol,
            period=period,
            n_simulations=n_simulations
        )

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Static Files (for built React app) ====================

# Check if frontend build exists
frontend_build_path = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.exists(frontend_build_path):
    # Serve static files
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_build_path, "static")), name="static")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve React app for all non-API routes"""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        index_path = os.path.join(frontend_build_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)

        raise HTTPException(status_code=404, detail="Not found")


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Saudi Stock AI Analyzer - API Server v3.0")
    print("Advanced LSTM Edition (BiLSTM + Multi-Head Attention)")
    print("=" * 60)
    print(f"\nStarting server on http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print("\nAvailable endpoints:")
    print("  GET  /api/stocks              - List available stocks")
    print("  GET  /api/analyze/{symbol}    - Full stock analysis")
    print("  GET  /api/chart/{symbol}      - Chart data")
    print("  GET  /api/predict/{symbol}    - ML prediction")
    print("  GET  /api/backtest/{symbol}   - Backtest results")
    print("  GET  /api/signals/history/{symbol} - Signal history")
    print("  GET  /api/compare             - Compare stocks")
    print("  GET  /api/risk/{symbol}       - Risk metrics")
    print("  POST /api/cache/clear         - Clear model cache")
    print("  GET  /api/health              - Health check")
    print("\n  Model Info:")
    print("  GET  /api/models              - List available models")
    print("\n  Market Rankings (NEW):")
    print("  GET  /api/market-rankings     - Top bullish/bearish stocks")
    print("  GET  /api/market-rankings/status - Cache status")
    print("  POST /api/market-rankings/refresh - Trigger refresh")
    print("\n  Position Manager:")
    print("  GET  /api/positions           - List all positions")
    print("  POST /api/positions           - Create new position")
    print("  GET  /api/positions/{id}      - Get single position")
    print("  PUT  /api/positions/{id}/close - Close position")
    print("  DELETE /api/positions/{id}    - Delete position")
    print("  POST /api/positions/from-signal/{symbol} - Create from signal")
    print("\n  Advanced Features:")
    print("  GET  /api/backtest/advanced/{symbol} - Monte Carlo backtest")
    print("\n  WebSocket (Real-time):")
    print("  WS   /ws/{symbol}             - Real-time stock updates")
    print("  WS   /ws/multi                - Multi-symbol subscriptions")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    uvicorn.run(
        "app:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=True
    )
