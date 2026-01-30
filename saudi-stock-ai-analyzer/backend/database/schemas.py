# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
Pydantic schemas for Position Manager
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class PositionResult(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    HOLD = "HOLD"


class PositionSource(str, Enum):
    MANUAL = "MANUAL"
    SIGNAL = "SIGNAL"


class PositionBase(BaseModel):
    """Base schema for position"""
    symbol: str = Field(..., description="Stock symbol (e.g., 2222)")
    entry_price: float = Field(..., gt=0, description="Entry price per share")
    quantity: float = Field(default=1.0, gt=0, description="Number of shares")
    notes: Optional[str] = Field(default=None, description="Optional notes")


class PositionCreate(PositionBase):
    """Schema for creating a new position"""
    source: PositionSource = Field(default=PositionSource.MANUAL, description="Position source")
    signal_id: Optional[str] = Field(default=None, description="Signal ID if created from signal")


class PositionUpdate(BaseModel):
    """Schema for updating a position"""
    current_price: Optional[float] = Field(default=None, gt=0, description="Current price")
    quantity: Optional[float] = Field(default=None, gt=0, description="Number of shares")
    notes: Optional[str] = Field(default=None, description="Notes")


class PositionClose(BaseModel):
    """Schema for closing a position"""
    exit_price: float = Field(..., gt=0, description="Exit price per share")


class Position(PositionBase):
    """Full position schema with all fields"""
    id: int
    entry_date: str
    current_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    source: PositionSource = PositionSource.MANUAL
    signal_id: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    result: PositionResult = PositionResult.HOLD
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class PositionSummary(BaseModel):
    """Summary statistics for positions"""
    total_positions: int = 0
    open_positions: int = 0
    closed_positions: int = 0
    total_pnl: float = 0.0
    total_pnl_percentage: float = 0.0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
