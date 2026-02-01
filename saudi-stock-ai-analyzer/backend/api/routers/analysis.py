from fastapi import APIRouter, HTTPException, Query
from main import StockAnalyzer

router = APIRouter(prefix="/api", tags=["analysis"])
analyzer = StockAnalyzer()

@router.get("/analyze/{symbol}")
async def analyze_stock(
    symbol: str,
    period: str = Query(default="6mo", description="Data period (1mo, 3mo, 6mo, 1y, 2y)"),
    train_model: bool = Query(default=True, description="Whether to train ML model"),
    force_retrain: bool = Query(default=False, description="Force model retraining")
):
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
    try:
        limit = min(limit, 100)
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
    try:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

        if len(symbol_list) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 symbols")
        if len(symbol_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")

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
