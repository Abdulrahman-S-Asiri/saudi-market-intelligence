"""
Position Manager - CRUD operations for positions
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from .position_db import PositionDB, init_database
from .schemas import (
    Position, PositionCreate, PositionUpdate, PositionClose,
    PositionStatus, PositionResult, PositionSource, PositionSummary
)


class PositionManager:
    """Manages position CRUD operations with P&L calculations"""

    def __init__(self):
        # Initialize database
        init_database()
        self.db = PositionDB()

    def _row_to_position(self, row) -> Dict[str, Any]:
        """Convert database row to position dictionary"""
        if row is None:
            return None
        return {
            'id': row['id'],
            'symbol': row['symbol'],
            'entry_price': row['entry_price'],
            'quantity': row['quantity'],
            'entry_date': row['entry_date'],
            'current_price': row['current_price'],
            'status': row['status'],
            'exit_price': row['exit_price'],
            'exit_date': row['exit_date'],
            'source': row['source'],
            'signal_id': row['signal_id'],
            'pnl': row['pnl'],
            'pnl_percentage': row['pnl_percentage'],
            'result': row['result'],
            'notes': row['notes'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        }

    def _calculate_pnl(self, entry_price: float, current_price: float, quantity: float) -> tuple:
        """Calculate P&L and P&L percentage"""
        pnl = (current_price - entry_price) * quantity
        pnl_percentage = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        return round(pnl, 2), round(pnl_percentage, 2)

    def _determine_result(self, pnl: float) -> str:
        """Determine WIN/LOSS/HOLD based on P&L"""
        if pnl > 0:
            return PositionResult.WIN.value
        elif pnl < 0:
            return PositionResult.LOSS.value
        return PositionResult.HOLD.value

    def create_position(self, position_data: PositionCreate) -> Dict[str, Any]:
        """Create a new position"""
        now = datetime.now().isoformat()

        data = {
            'symbol': position_data.symbol,
            'entry_price': position_data.entry_price,
            'quantity': position_data.quantity,
            'entry_date': now,
            'current_price': position_data.entry_price,  # Initial current price = entry price
            'status': PositionStatus.OPEN.value,
            'source': position_data.source.value if isinstance(position_data.source, PositionSource) else position_data.source,
            'signal_id': position_data.signal_id,
            'pnl': 0.0,
            'pnl_percentage': 0.0,
            'result': PositionResult.HOLD.value,
            'notes': position_data.notes,
            'created_at': now,
            'updated_at': now
        }

        position_id = self.db.insert('positions', data)
        return self.get_position(position_id)

    def create_from_signal(self, symbol: str, entry_price: float, signal_id: str = None) -> Dict[str, Any]:
        """Create a position from a BUY signal"""
        position_data = PositionCreate(
            symbol=symbol,
            entry_price=entry_price,
            quantity=1.0,
            source=PositionSource.SIGNAL,
            signal_id=signal_id or f"signal_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        return self.create_position(position_data)

    def get_position(self, position_id: int) -> Optional[Dict[str, Any]]:
        """Get a single position by ID"""
        row = self.db.fetch_one(
            "SELECT * FROM positions WHERE id = ?",
            (position_id,)
        )
        return self._row_to_position(row)

    def get_all_positions(self, status: str = None) -> List[Dict[str, Any]]:
        """Get all positions, optionally filtered by status"""
        if status:
            rows = self.db.fetch_all(
                "SELECT * FROM positions WHERE status = ? ORDER BY created_at DESC",
                (status,)
            )
        else:
            rows = self.db.fetch_all(
                "SELECT * FROM positions ORDER BY created_at DESC"
            )
        return [self._row_to_position(row) for row in rows]

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        return self.get_all_positions(status=PositionStatus.OPEN.value)

    def get_closed_positions(self) -> List[Dict[str, Any]]:
        """Get all closed positions"""
        return self.get_all_positions(status=PositionStatus.CLOSED.value)

    def update_position(self, position_id: int, update_data: PositionUpdate) -> Optional[Dict[str, Any]]:
        """Update a position"""
        position = self.get_position(position_id)
        if not position:
            return None

        now = datetime.now().isoformat()
        data = {'updated_at': now}

        if update_data.current_price is not None:
            data['current_price'] = update_data.current_price
            # Recalculate P&L
            pnl, pnl_percentage = self._calculate_pnl(
                position['entry_price'],
                update_data.current_price,
                position['quantity']
            )
            data['pnl'] = pnl
            data['pnl_percentage'] = pnl_percentage
            data['result'] = self._determine_result(pnl)

        if update_data.quantity is not None:
            data['quantity'] = update_data.quantity
            # Recalculate P&L with new quantity
            current_price = data.get('current_price', position['current_price'])
            if current_price:
                pnl, pnl_percentage = self._calculate_pnl(
                    position['entry_price'],
                    current_price,
                    update_data.quantity
                )
                data['pnl'] = pnl
                data['pnl_percentage'] = pnl_percentage

        if update_data.notes is not None:
            data['notes'] = update_data.notes

        self.db.update('positions', data, 'id = ?', (position_id,))
        return self.get_position(position_id)

    def update_current_price(self, position_id: int, current_price: float) -> Optional[Dict[str, Any]]:
        """Update the current price and recalculate P&L"""
        return self.update_position(position_id, PositionUpdate(current_price=current_price))

    def update_prices_for_symbol(self, symbol: str, current_price: float) -> int:
        """Update current price for all open positions of a symbol"""
        positions = self.db.fetch_all(
            "SELECT id, entry_price, quantity FROM positions WHERE symbol = ? AND status = ?",
            (symbol, PositionStatus.OPEN.value)
        )

        updated_count = 0
        now = datetime.now().isoformat()

        for pos in positions:
            pnl, pnl_percentage = self._calculate_pnl(
                pos['entry_price'],
                current_price,
                pos['quantity']
            )

            data = {
                'current_price': current_price,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'result': self._determine_result(pnl),
                'updated_at': now
            }

            self.db.update('positions', data, 'id = ?', (pos['id'],))
            updated_count += 1

        return updated_count

    def close_position(self, position_id: int, close_data: PositionClose) -> Optional[Dict[str, Any]]:
        """Close a position with exit price"""
        position = self.get_position(position_id)
        if not position:
            return None

        if position['status'] == PositionStatus.CLOSED.value:
            return position  # Already closed

        now = datetime.now().isoformat()

        # Calculate final P&L
        pnl, pnl_percentage = self._calculate_pnl(
            position['entry_price'],
            close_data.exit_price,
            position['quantity']
        )

        data = {
            'status': PositionStatus.CLOSED.value,
            'exit_price': close_data.exit_price,
            'exit_date': now,
            'current_price': close_data.exit_price,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'result': self._determine_result(pnl),
            'updated_at': now
        }

        self.db.update('positions', data, 'id = ?', (position_id,))
        return self.get_position(position_id)

    def delete_position(self, position_id: int) -> bool:
        """Delete a position"""
        affected = self.db.delete('positions', 'id = ?', (position_id,))
        return affected > 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all positions"""
        all_positions = self.get_all_positions()
        open_positions = [p for p in all_positions if p['status'] == PositionStatus.OPEN.value]
        closed_positions = [p for p in all_positions if p['status'] == PositionStatus.CLOSED.value]

        # Calculate totals from closed positions
        total_pnl = sum(p['pnl'] or 0 for p in closed_positions)
        wins = len([p for p in closed_positions if p['result'] == PositionResult.WIN.value])
        losses = len([p for p in closed_positions if p['result'] == PositionResult.LOSS.value])

        # Open positions unrealized P&L
        unrealized_pnl = sum(p['pnl'] or 0 for p in open_positions)

        win_rate = (wins / len(closed_positions) * 100) if closed_positions else 0

        return {
            'total_positions': len(all_positions),
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'total_pnl': round(total_pnl, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1)
        }

    def get_positions_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all positions for a specific symbol"""
        rows = self.db.fetch_all(
            "SELECT * FROM positions WHERE symbol = ? ORDER BY created_at DESC",
            (symbol,)
        )
        return [self._row_to_position(row) for row in rows]
