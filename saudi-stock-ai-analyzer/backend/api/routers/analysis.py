from fastapi import APIRouter, HTTPException, Query
from main import StockAnalyzer
from utils.validators import (
    validate_symbol, validate_period, validate_symbols_list,
    validate_confidence, validate_hold_period, validate_limit
)

router = APIRouter(prefix="/api", tags=["analysis"])
analyzer = StockAnalyzer()

@router.get("/analyze/{symbol}")
async def analyze_stock(
    symbol: str,
    period: str = Query(default="6mo", description="Data period (1mo, 3mo, 6mo, 1y, 2y)"),
    train_model: bool = Query(default=True, description="Whether to train ML model"),
    force_retrain: bool = Query(default=False, description="Force model retraining")
):
    # Validate inputs
    validate_symbol(symbol)
    validate_period(period)

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

@router.get("/chart/{symbol}")
async def get_chart_data(
    symbol: str,
    period: str = Query(default="6mo", description="Data period")
):
    # Validate inputs
    validate_symbol(symbol)
    validate_period(period)

    try:
        result = analyzer.get_chart_data(symbol, period=period)

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/{symbol}")
async def predict_stock(
    symbol: str,
    period: str = Query(default="1y", description="Training data period")
):
    # Validate inputs
    validate_symbol(symbol)
    validate_period(period)

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

@router.get("/backtest/{symbol}")
async def backtest_stock(
    symbol: str,
    period: str = Query(default="1y", description="Backtest period (6mo, 1y, 2y)"),
    min_confidence: float = Query(default=60, description="Minimum confidence to take trades (0-100)"),
    hold_period: int = Query(default=5, description="Default holding period in days")
):
    # Validate inputs
    validate_symbol(symbol)
    validate_period(period)
    validate_confidence(min_confidence)
    validate_hold_period(hold_period)

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

@router.get("/signals/history/{symbol}")
async def get_signal_history(
    symbol: str,
    limit: int = Query(default=20, description="Number of signals to return (max 100)")
):
    # Validate inputs
    validate_symbol(symbol)
    validate_limit(limit, max_limit=100)

    try:
        signals = analyzer.get_signal_history(symbol, limit=limit)

        return {
            "symbol": symbol,
            "count": len(signals),
            "signals": signals
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compare")
async def compare_stocks(
    symbols: str = Query(description="Comma-separated list of stock symbols"),
    period: str = Query(default="6mo", description="Data period for comparison")
):
    # Parse and validate inputs
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

    if len(symbol_list) < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 symbols")
    if len(symbol_list) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")

    validate_symbols_list(symbol_list)
    validate_period(period)

    try:
        result = analyzer.compare_stocks(symbol_list, period=period)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/{symbol}")
async def get_risk_metrics(
    symbol: str,
    period: str = Query(default="1y", description="Data period for risk calculation")
):
    # Validate inputs
    validate_symbol(symbol)
    validate_period(period)

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


# ==================== Market Rankings Logic (Restored) ====================

from typing import Optional, List, Dict
from datetime import datetime
import threading
from fastapi import BackgroundTasks

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
            sma_50 = indicators.get('sma_50', 0)
            
            signal_action = signal.get('action', 'HOLD')
            signal_confidence = signal.get('confidence', 50)
            volatility = performance.get('volatility', 15)
            total_return = performance.get('total_return', 0)
            price = signal.get('price', 0)

            # RSI component
            rsi_component = (50 - rsi) / 40
            rsi_component = max(-1, min(1, rsi_component))

            # MACD component
            macd_component = 0
            if macd != 0:
                macd_component = 0.5 if macd > macd_signal else -0.5

            # Recent momentum
            momentum_component = max(-0.5, min(0.5, total_return / 20))

            # Trend component (Price relative to SMA 50)
            trend_component = 0
            if sma_50 > 0 and price > 0:
                trend_component = 0.5 if price > sma_50 else -0.5

            # Combined score (Refined weights for better accuracy)
            # RSI (Reversion) 40%, MACD (Trend) 30%, Momentum 20%, Long Trend 10%
            combined_score = (rsi_component * 0.4) + (macd_component * 0.3) + (momentum_component * 0.2) + (trend_component * 0.1)

            # Predicted percentage change
            # Dampen the volatility factor slightly to avoid unrealistic predictions
            volatility_factor = min(volatility / 100 * 1.5, 0.05)
            predicted_change = combined_score * volatility_factor * 100

            # MC Dropout Confidence
            confidence = 50.0
            ml_uncertainty = None
            ml_variance = None

            if use_ml_confidence and ml_prediction:
                ml_conf = ml_prediction.get('confidence', 50)
                ml_uncertainty = ml_prediction.get('uncertainty', 0.05)
                ml_variance = ml_prediction.get('variance', 0.0025)

                if ml_conf is not None:
                    confidence = float(ml_conf)
                
                ml_pred_value = ml_prediction.get('prediction', 0)
                if ml_pred_value != 0:
                    predicted_change = 0.6 * (ml_pred_value * 100) + 0.4 * predicted_change
            else:
                indicator_strength = abs(rsi_component) * 50 + abs(macd_component) * 30
                confidence = min(max(indicator_strength + 20, 25), 95)

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
                "confidence_method": "mc_dropout" if use_ml_confidence else "heuristic",
                "uncertainty": round(ml_uncertainty, 6) if use_ml_confidence and ml_uncertainty else None,
                "variance": round(ml_variance, 8) if use_ml_confidence and ml_variance else None
            }

            results.append(result)

        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
            continue

    # Sort
    sorted_by_gain = sorted(results, key=lambda x: x['predicted_change'], reverse=True)
    top_bullish = sorted_by_gain[:5]
    
    sorted_by_loss = sorted(results, key=lambda x: x['predicted_change'])
    top_bearish = sorted_by_loss[:5]

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

@router.get("/market-rankings")
async def get_market_rankings(
    background_tasks: BackgroundTasks,
    force_refresh: bool = Query(default=False, description="Force refresh rankings (bypass cache)"),
    quick_scan: bool = Query(default=True, description="Scan only top 30 stocks (faster)"),
    use_ml: bool = Query(default=True, description="Use Monte Carlo Dropout")
):
    """
    Get top projected gainers and losers.
    Uses caching to return immediate results while updating in background if needed.
    """
    # Simply return what we have if cache is valid and no force refresh
    if not force_refresh and market_rankings_cache.is_cache_valid():
        return market_rankings_cache.get_cached_rankings()

    # Determine stocks to scan
    stocks_to_scan = TOP_STOCKS_FOR_RANKING if quick_scan else analyzer.get_available_stocks()

    # If updating is already in progress, return cached (even if expired) or error
    if market_rankings_cache.is_updating():
        cached = market_rankings_cache.get_cached_rankings()
        if cached:
            return {**cached, "status": "updating_in_background"}
        return {"status": "calculating", "message": "Market scan in progress, please check back in a minute"}

    # Define update task
    def update_task_logic():
        market_rankings_cache.set_updating(True)
        try:
            rankings = scan_stocks_for_rankings(stocks_to_scan, analyzer, use_ml_confidence=use_ml)
            market_rankings_cache.update_rankings(rankings)
        except Exception as e:
            print(f"Error updating rankings: {e}")
        finally:
            market_rankings_cache.set_updating(False)

    # Trigger update
    if force_refresh or not market_rankings_cache.get_cached_rankings():
        # If forced or no cache, do it synchronously (slow but needed for first load)
        # Or better: do it in background and return 'processing' status
        # For better UX, let's do sync if it's the very first time, otherwise async
        if not market_rankings_cache.get_cached_rankings():
             market_rankings_cache.set_updating(True)
             rankings = scan_stocks_for_rankings(stocks_to_scan, analyzer, use_ml_confidence=use_ml)
             market_rankings_cache.update_rankings(rankings)
             return rankings
        
        background_tasks.add_task(update_task_logic)
        return {**market_rankings_cache.get_cached_rankings(), "status": "refreshing_in_background"}

    return market_rankings_cache.get_cached_rankings()
