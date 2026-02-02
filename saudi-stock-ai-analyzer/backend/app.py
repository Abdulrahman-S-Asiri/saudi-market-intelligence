# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
FastAPI server for Saudi Stock AI Analyzer
Provides REST API endpoints for the React frontend
Uses Advanced LSTM model (BiLSTM + Multi-Head Attention)
"""

import sys
import os
import asyncio
import time
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import API_CONFIG
from api.routers import stocks, analysis, portfolio
from main import StockAnalyzer

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

# Include Routers
app.include_router(stocks.router)
app.include_router(analysis.router)
app.include_router(portfolio.router)

# Initialize analyzer for root endpoints
analyzer = StockAnalyzer()

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
            "analyze": "/api/analyze/{symbol}",
            "positions": "/api/positions"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "saudi-stock-ai-analyzer",
        "version": "3.0.0"
    }

@app.post("/api/cache/clear")
async def clear_cache(
    symbol: str = Query(default=None, description="Symbol to clear (or all if not specified)")
):
    try:
        analyzer.clear_model_cache(symbol)
        return {
            "status": "success",
            "message": f"Cache cleared for {'all symbols' if symbol is None else symbol}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
