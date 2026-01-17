"""
FastAPI server for Saudi Stock AI Analyzer
Provides REST API endpoints for the React frontend
"""

import sys
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import StockAnalyzer
from utils.config import API_CONFIG, SAUDI_STOCKS

# Initialize FastAPI app
app = FastAPI(
    title="Saudi Stock AI Analyzer API",
    description="AI-powered stock analysis for Saudi Arabian market (Tadawul)",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer (singleton)
analyzer = StockAnalyzer(use_ensemble=True)


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Saudi Stock AI Analyzer API",
        "version": "2.0.0",
        "endpoints": {
            "stocks": "/api/stocks",
            "analyze": "/api/analyze/{symbol}",
            "chart": "/api/chart/{symbol}",
            "predict": "/api/predict/{symbol}",
            "backtest": "/api/backtest/{symbol}",
            "signals_history": "/api/signals/history/{symbol}",
            "compare": "/api/compare",
            "risk": "/api/risk/{symbol}",
            "health": "/api/health"
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


@app.get("/api/analyze/{symbol}")
async def analyze_stock(
    symbol: str,
    period: str = Query(default="6mo", description="Data period (1mo, 3mo, 6mo, 1y, 2y)"),
    train_model: bool = Query(default=True, description="Whether to train ML model"),
    force_retrain: bool = Query(default=False, description="Force model retraining")
):
    """
    Perform complete analysis on a stock

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
    Get ML prediction for a stock

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
        "version": "2.0.0",
        "features": {
            "ensemble_model": analyzer.use_ensemble,
            "model_caching": True,
            "backtesting": True,
            "risk_metrics": True
        }
    }


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
    print("Saudi Stock AI Analyzer - API Server v2.0")
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
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    uvicorn.run(
        "app:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=True
    )
