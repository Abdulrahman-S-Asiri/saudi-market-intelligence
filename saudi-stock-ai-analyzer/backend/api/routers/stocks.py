from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
from utils.config import SAUDI_STOCKS, SECTORS
from main import StockAnalyzer

router = APIRouter(prefix="/api", tags=["stocks"])

# Use the singleton analyzer from app state (can be injected, but for now we import or re-instantiate)
# Better pattern: Dependency injection. For now, let's assume we import the singleton wrapper or similar.
# To keep it simple and consistent with existing app.py, we'll instantiate/import it.
# Ideally, we should move the singleton creation to a shared module.

from main import StockAnalyzer
analyzer = StockAnalyzer()

@router.get("/stocks")
async def get_stocks():
    """Get list of available Saudi stocks"""
    stocks = analyzer.get_available_stocks()
    return {
        "count": len(stocks),
        "stocks": stocks
    }

@router.get("/sectors")
async def get_sectors():
    """Get all available sectors with their stocks"""
    return {
        "count": len(SECTORS),
        "sectors": SECTORS
    }

@router.get("/stocks/sector/{sector}")
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
