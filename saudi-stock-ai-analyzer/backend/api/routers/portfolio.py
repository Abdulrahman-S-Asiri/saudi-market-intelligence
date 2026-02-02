from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List
from database import PositionManager, PositionCreate, PositionUpdate, PositionClose
from main import StockAnalyzer
from utils.validators import (
    validate_symbol, validate_status,
    validate_amount, validate_stop_loss_pct, validate_take_profit_pct
)

router = APIRouter(prefix="/api/positions", tags=["positions"])
position_manager = PositionManager()
analyzer = StockAnalyzer()

@router.get("")
async def list_positions(
    status: Optional[str] = Query(None, description="Filter by status (OPEN, CLOSED)"),
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    # Validate inputs if provided
    if status:
        validate_status(status)
    if symbol:
        validate_symbol(symbol)

    if symbol:
        positions = position_manager.get_positions_by_symbol(symbol)
        if status:
            positions = [p for p in positions if p['status'] == status]
    else:
        positions = position_manager.get_all_positions(status=status)
    return positions

@router.post("")
async def create_position(position: PositionCreate):
    try:
        new_position = position_manager.create_position(position)
        return new_position
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_portfolio_summary():
    return position_manager.get_summary()

@router.get("/{position_id}")
async def get_position(position_id: str):
    position = position_manager.get_position(position_id)
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")
    return position

@router.put("/{position_id}")
async def update_position(position_id: str, update: PositionUpdate):
    try:
        position = position_manager.update_position(position_id, update)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        return position
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{position_id}/close")
async def close_position(position_id: str, close_data: PositionClose):
    try:
        position = position_manager.close_position(
            position_id, 
            price=close_data.exit_price, 
            exit_date=close_data.exit_date,
            reason=close_data.exit_reason
        )
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        return position
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{position_id}")
async def delete_position(position_id: str):
    success = position_manager.delete_position(position_id)
    if not success:
        raise HTTPException(status_code=404, detail="Position not found")
    return {"status": "success", "message": "Position deleted"}

@router.post("/from-signal/{symbol}")
async def create_position_from_signal(
    symbol: str,
    amount: float = Query(default=10000, description="Investment amount"),
    stop_loss_pct: float = Query(default=5.0, description="Stop loss percentage"),
    take_profit_pct: float = Query(default=10.0, description="Take profit percentage")
):
    # Validate inputs
    validate_symbol(symbol)
    validate_amount(amount)
    validate_stop_loss_pct(stop_loss_pct)
    validate_take_profit_pct(take_profit_pct)

    try:
        # Get latest analysis
        analysis = analyzer.analyze_stock(symbol, period="3mo", train_model=False)
        
        if 'error' in analysis:
            raise HTTPException(status_code=400, detail=f"Could not analyze stock: {analysis['error']}")
            
        signal = analysis.get('signal', {})
        price = signal.get('price')
        
        if not price:
            raise HTTPException(status_code=400, detail="Could not determine current price")
            
        action = signal.get('action', 'HOLD')
        if action == 'HOLD':
             raise HTTPException(status_code=400, detail="Current signal is HOLD, cannot open position")
             
        position_type = "LONG" if action == "BUY" else "SHORT" # Saudi market mostly long, but simplified
        
        # Calculate quantity
        quantity = amount / price
        
        # Create position object
        position_data = PositionCreate(
            symbol=symbol,
            entry_price=price,
            quantity=quantity,
            position_type=position_type,
            stop_loss=price * (1 - stop_loss_pct/100) if position_type == "LONG" else price * (1 + stop_loss_pct/100),
            take_profit=price * (1 + take_profit_pct/100) if position_type == "LONG" else price * (1 - take_profit_pct/100),
            notes=f"Auto-created from {action} signal. Confidence: {signal.get('confidence', 0)}%"
        )
        
        new_position = position_manager.create_position(position_data)
        return new_position
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
