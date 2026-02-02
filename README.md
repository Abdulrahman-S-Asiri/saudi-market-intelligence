# Saudi Stock AI Analyzer (TASI Intelligence)

<div align="center">

![Version](https://img.shields.io/badge/version-3.0.0-blue.svg?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![Vite](https://img.shields.io/badge/Vite-7-646CFF?style=for-the-badge&logo=vite)

**AI-Powered Market Intelligence for the Saudi Exchange (Tadawul)**

[Features](#features) | [Pages](#pages) | [Tech Stack](#tech-stack) | [Getting Started](#getting-started) | [API](#api-endpoints)

</div>

---

## Overview

**Saudi Stock AI Analyzer** is a financial analytics platform that brings institutional-grade insights to the Saudi Market (TASI). It combines **Deep Learning (BiLSTM + Multi-Head Attention)** with **Monte Carlo Dropout** for calibrated confidence estimation, providing actionable trade signals with quantified uncertainty.

The platform covers **200+ Saudi stocks** across all TASI sectors including Energy, Banks, Materials, Telecom, Healthcare, REITs, and more.

## Features

### AI-Powered Analysis
- **BiLSTM + Multi-Head Attention**: Neural network architecture capturing long-term market dependencies
- **Monte Carlo Dropout**: Scientifically calibrated confidence scores (25-95%) through Bayesian uncertainty estimation
- **40+ Technical Features**: RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, ADX, and more
- **Market Regime Detection**: Automatic classification of Bull/Bear/Sideways conditions

### Trading Signals
- **BUY/SELL/HOLD Recommendations**: Clear actionable signals with confidence levels
- **Entry/Exit Targets**: Suggested price points for trade execution
- **Stop Loss & Take Profit**: Risk management levels calculated per signal
- **Signal History**: Track past signals and performance

### Portfolio Management
- **Position Tracking**: Open and manage positions directly from signals
- **P/L Calculation**: Real-time profit/loss tracking
- **Risk Metrics**: VaR, Sharpe Ratio, Sortino Ratio, Max Drawdown

### Market Scanner
- **Top Movers**: Ranked stocks by predicted upside/downside
- **Bulk Analysis**: Scan 30+ key stocks for opportunities
- **Sector Coverage**: All major TASI sectors represented

## Pages

The application includes 4 main pages accessible via navigation:

| Page | Description |
|------|-------------|
| **Dashboard** | Main trading view with interactive chart, AI signals, timeframe selector, and quick trade execution |
| **Analysis** | Detailed stock analysis with technical indicators, ML predictions, risk metrics, and signal reasons |
| **Scanner** | Market-wide scanner showing top bullish and bearish signals with confidence levels |
| **Portfolio** | Position management with open/closed positions, P/L tracking, and trade history |

## Tech Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18 | UI Framework |
| Vite 7 | Build Tool & Dev Server |
| React Router 7 | Client-side Navigation |
| TanStack Query 5 | Data Fetching & Caching |
| Tailwind CSS 3 | Styling |
| Lightweight Charts | TradingView-style Charts |
| Heroicons | Icons |
| Axios | HTTP Client |
| Vitest | Testing |

### Backend
| Technology | Purpose |
|------------|---------|
| FastAPI | Async API Framework |
| PyTorch | Deep Learning (BiLSTM) |
| Pandas/NumPy | Data Processing |
| Scikit-learn | ML Utilities |
| yfinance | Market Data |
| SQLite | Position Storage |
| Pytest | Testing |

## Project Structure

```
saudi-stock-ai-analyzer/
├── backend/
│   ├── api/routers/          # API route handlers
│   │   ├── analysis.py       # Stock analysis endpoints
│   │   └── portfolio.py      # Position management endpoints
│   ├── backtest/             # Backtesting engine
│   ├── data/                 # Data loaders & signal storage
│   ├── database/             # Position database
│   ├── models/               # ML models & market regime
│   ├── strategy/             # Trading strategy & risk management
│   ├── utils/                # Config, validators, logging
│   ├── app.py                # FastAPI application
│   └── main.py               # Stock analyzer core
│
├── frontend/
│   ├── src/
│   │   ├── api/              # API client
│   │   ├── components/       # React components
│   │   ├── hooks/            # Custom React hooks
│   │   ├── pages/            # Page components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Analysis.jsx
│   │   │   ├── Scanner.jsx
│   │   │   └── Portfolio.jsx
│   │   ├── styles/           # CSS styles
│   │   ├── App.jsx           # Main app with routes
│   │   └── index.jsx         # Entry point
│   └── package.json
│
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend Setup
```bash
cd saudi-stock-ai-analyzer/backend
pip install -r requirements.txt
uvicorn app:app --reload
```
Backend runs at `http://localhost:8000`

### Frontend Setup
```bash
cd saudi-stock-ai-analyzer/frontend
npm install
npm run dev
```
Frontend runs at `http://localhost:5173`

## API Endpoints

### Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stocks` | GET | List all available stocks |
| `/api/analyze/{symbol}` | GET | Full stock analysis with ML prediction |
| `/api/chart/{symbol}` | GET | OHLCV chart data |
| `/api/predict/{symbol}` | GET | ML prediction only |
| `/api/backtest/{symbol}` | GET | Run strategy backtest |
| `/api/signals/history/{symbol}` | GET | Historical signals |
| `/api/compare` | GET | Compare multiple stocks |
| `/api/risk/{symbol}` | GET | Risk metrics |
| `/api/market-rankings` | GET | Top bullish/bearish stocks |

### Portfolio
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/positions` | GET | List all positions |
| `/api/positions` | POST | Create new position |
| `/api/positions/summary` | GET | Portfolio summary |
| `/api/positions/{id}` | GET | Get position by ID |
| `/api/positions/{id}` | PUT | Update position |
| `/api/positions/{id}/close` | PUT | Close position |
| `/api/positions/{id}` | DELETE | Delete position |
| `/api/positions/from-signal/{symbol}` | POST | Create position from signal |

### Input Validation
All endpoints validate:
- **Symbol**: Must be a valid TASI stock symbol
- **Period**: Must be one of `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`
- **Confidence**: 0-100 range
- **Amount**: > 0 and <= 1,000,000
- **Stop Loss %**: > 0 and <= 50
- **Take Profit %**: > 0 and <= 500

## Testing

```bash
# Backend Tests
cd saudi-stock-ai-analyzer/backend
pytest

# Frontend Tests
cd saudi-stock-ai-analyzer/frontend
npm test
```

## Supported Stocks

The platform supports **200+ stocks** across all TASI sectors:

- **Energy**: Saudi Aramco (2222), Petro Rabigh, Bahri
- **Banks**: Al Rajhi (1120), SNB (1182), Alinma (1180), Riyad Bank
- **Materials**: SABIC (2010), Ma'aden (1211), Saudi Kayan
- **Telecom**: STC (7010), Mobily (7020), Zain (7030)
- **Healthcare**: Dr. Sulaiman Al Habib (4013), Mouwasat
- **Utilities**: Saudi Electricity (5110), ACWA Power
- **Consumer**: Almarai (2280), Jarir (4190), Al Othaim
- **REITs**: Riyad REIT, Jadwa REIT, and more
- **Insurance**: Bupa Arabia, Tawuniya
- **Software**: Elm (7201), Solutions by STC

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built by Abdulrahman Asiri</sub>
</div>
