"""
Input validation utilities for Saudi Stock AI Analyzer API
"""

from fastapi import HTTPException
from typing import List
from utils.config import SAUDI_STOCKS

# Valid periods for stock data
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"}

# Valid position statuses
VALID_STATUSES = {"OPEN", "CLOSED"}


def validate_symbol(symbol: str) -> str:
    """
    Validate that a symbol exists in the SAUDI_STOCKS list.

    Args:
        symbol: Stock symbol to validate

    Returns:
        The validated symbol

    Raises:
        HTTPException: If symbol is not valid
    """
    if symbol not in SAUDI_STOCKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid symbol: {symbol}. Must be a valid Saudi stock symbol."
        )
    return symbol


def validate_period(period: str) -> str:
    """
    Validate that a period is one of the allowed values.

    Args:
        period: Period string to validate

    Returns:
        The validated period

    Raises:
        HTTPException: If period is not valid
    """
    if period not in VALID_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period: {period}. Must be one of: {', '.join(sorted(VALID_PERIODS))}"
        )
    return period


def validate_symbols_list(symbols: List[str]) -> List[str]:
    """
    Validate a list of symbols.

    Args:
        symbols: List of stock symbols to validate

    Returns:
        The validated list of symbols

    Raises:
        HTTPException: If any symbol is not valid
    """
    invalid_symbols = [s for s in symbols if s not in SAUDI_STOCKS]
    if invalid_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid symbols: {', '.join(invalid_symbols)}. Must be valid Saudi stock symbols."
        )
    return symbols


def validate_confidence(confidence: float) -> float:
    """
    Validate that confidence is within valid range (0-100).

    Args:
        confidence: Confidence value to validate

    Returns:
        The validated confidence

    Raises:
        HTTPException: If confidence is out of range
    """
    if confidence < 0 or confidence > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid confidence: {confidence}. Must be between 0 and 100."
        )
    return confidence


def validate_hold_period(hold_period: int) -> int:
    """
    Validate that hold period is within valid range (1-100).

    Args:
        hold_period: Hold period in days to validate

    Returns:
        The validated hold period

    Raises:
        HTTPException: If hold period is out of range
    """
    if hold_period < 1 or hold_period > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hold_period: {hold_period}. Must be between 1 and 100 days."
        )
    return hold_period


def validate_limit(limit: int, max_limit: int = 100) -> int:
    """
    Validate that limit is within valid range (1-max_limit).

    Args:
        limit: Limit value to validate
        max_limit: Maximum allowed limit

    Returns:
        The validated limit

    Raises:
        HTTPException: If limit is out of range
    """
    if limit < 1 or limit > max_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid limit: {limit}. Must be between 1 and {max_limit}."
        )
    return limit


def validate_status(status: str) -> str:
    """
    Validate that status is a valid position status.

    Args:
        status: Status string to validate

    Returns:
        The validated status

    Raises:
        HTTPException: If status is not valid
    """
    if status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {status}. Must be one of: {', '.join(VALID_STATUSES)}"
        )
    return status


def validate_amount(amount: float) -> float:
    """
    Validate that amount is within valid range (>0 and <=1,000,000).

    Args:
        amount: Investment amount to validate

    Returns:
        The validated amount

    Raises:
        HTTPException: If amount is out of range
    """
    if amount <= 0 or amount > 1000000:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid amount: {amount}. Must be greater than 0 and at most 1,000,000."
        )
    return amount


def validate_stop_loss_pct(pct: float) -> float:
    """
    Validate that stop loss percentage is within valid range (>0 and <=50).

    Args:
        pct: Stop loss percentage to validate

    Returns:
        The validated percentage

    Raises:
        HTTPException: If percentage is out of range
    """
    if pct <= 0 or pct > 50:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stop_loss_pct: {pct}. Must be greater than 0 and at most 50."
        )
    return pct


def validate_take_profit_pct(pct: float) -> float:
    """
    Validate that take profit percentage is within valid range (>0 and <=500).

    Args:
        pct: Take profit percentage to validate

    Returns:
        The validated percentage

    Raises:
        HTTPException: If percentage is out of range
    """
    if pct <= 0 or pct > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid take_profit_pct: {pct}. Must be greater than 0 and at most 500."
        )
    return pct
