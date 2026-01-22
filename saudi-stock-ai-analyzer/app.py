"""
FastAPI server for Saudi Stock AI Analyzer
Provides REST API endpoints for the React frontend
Uses Advanced LSTM model (BiLSTM with Multi-Head Attention)
"""

import sys
import os
import asyncio
import json
from typing import Optional, List, Dict, Set
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import StockAnalyzer
from utils.config import API_CONFIG, SAUDI_STOCKS, SECTORS
from database import PositionManager, PositionCreate, PositionUpdate, PositionClose

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
            "position_manager": True
        }
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
