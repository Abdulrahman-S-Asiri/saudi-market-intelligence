<div align="center">

# Saudi Stock AI Analyzer

### Institutional-Grade Market Intelligence for Tadawul (TASI)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br />

<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/react/react-original.svg" width="40" />&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="40" />&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytorch/pytorch-original.svg" width="40" />&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tailwindcss/tailwindcss-original.svg" width="40" />

<br />
<br />

A production-ready financial analytics platform combining **BiLSTM neural networks** with **Monte Carlo uncertainty estimation** to deliver actionable trading signals for 200+ Saudi stocks.

[Getting Started](#getting-started) · [Documentation](#api-reference) · [Architecture](#architecture)

</div>

<br />

---

<br />

## Overview

Saudi Stock AI Analyzer provides hedge-fund quality analysis for the Saudi Exchange. The platform processes real-time market data through a deep learning pipeline that generates calibrated trading signals with confidence intervals.

**Core Capabilities:**

| Capability | Description |
|:-----------|:------------|
| **Deep Learning Signals** | BiLSTM + Multi-Head Attention architecture trained on 40+ technical features |
| **Uncertainty Quantification** | Monte Carlo Dropout provides calibrated confidence scores (25-95%) |
| **Position Management** | Full portfolio tracking with P/L, risk metrics, and trade execution |
| **Market Scanning** | Real-time ranking of stocks by predicted movement and signal strength |

<br />

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  FRONTEND                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Dashboard  │  │  Analysis   │  │   Scanner   │  │  Portfolio  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         └─────────────────┴─────────────────┴─────────────────┘             │
│                                    │                                         │
│                         React + TanStack Query                               │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ REST API
┌─────────────────────────────────────┴───────────────────────────────────────┐
│                                  BACKEND                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         FastAPI Gateway                               │   │
│  │              /analyze  /predict  /positions  /market-rankings         │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Analyzer   │  │  ML Engine   │  │   Strategy   │  │   Backtest   │    │
│  │              │◄─┤  BiLSTM +    │  │  Risk Mgmt   │  │    Engine    │    │
│  │  Indicators  │  │  Attention   │  │  Regime Det  │  │  Monte Carlo │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                     │                                        │
│  ┌──────────────────────────────────┴───────────────────────────────────┐   │
│  │                         Data Layer                                    │   │
│  │                  yfinance API  ←→  SQLite Positions                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

<br />

## Tech Stack

<table>
<tr>
<td width="50%" valign="top">

### Frontend
- **React 18** — Component framework
- **Vite 7** — Build tooling
- **React Router 7** — Navigation
- **TanStack Query 5** — Data management
- **Tailwind CSS** — Styling
- **Lightweight Charts** — Financial charts
- **Vitest** — Testing

</td>
<td width="50%" valign="top">

### Backend
- **FastAPI** — Async API framework
- **PyTorch** — Neural network engine
- **Pandas / NumPy** — Data processing
- **Scikit-learn** — ML utilities
- **yfinance** — Market data feed
- **SQLite** — Position storage
- **Pytest** — Testing

</td>
</tr>
</table>

<br />

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Abdulrahman-S-Asiri/saudi-market-intelligence.git
cd saudi-market-intelligence/saudi-stock-ai-analyzer
```

**2. Start the backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**3. Start the frontend**
```bash
cd frontend
npm install
npm run dev
```

**4. Access the application**
```
Frontend:  http://localhost:5173
Backend:   http://localhost:8000
API Docs:  http://localhost:8000/docs
```

<br />

## API Reference

### Analysis Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/api/stocks` | List all available stocks |
| `GET` | `/api/analyze/{symbol}` | Full analysis with ML prediction |
| `GET` | `/api/chart/{symbol}` | OHLCV price data |
| `GET` | `/api/predict/{symbol}` | ML prediction only |
| `GET` | `/api/backtest/{symbol}` | Strategy backtest results |
| `GET` | `/api/signals/history/{symbol}` | Historical signals |
| `GET` | `/api/compare?symbols=X,Y` | Multi-stock comparison |
| `GET` | `/api/risk/{symbol}` | Risk metrics (VaR, Sharpe) |
| `GET` | `/api/market-rankings` | Top movers scan |

### Portfolio Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/api/positions` | List positions |
| `POST` | `/api/positions` | Create position |
| `GET` | `/api/positions/summary` | Portfolio summary |
| `PUT` | `/api/positions/{id}/close` | Close position |
| `DELETE` | `/api/positions/{id}` | Delete position |
| `POST` | `/api/positions/from-signal/{symbol}` | Create from signal |

### Query Parameters

| Parameter | Valid Values | Default |
|:----------|:-------------|:--------|
| `period` | `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y` | `6mo` |
| `train_model` | `true`, `false` | `true` |
| `min_confidence` | `0-100` | `60` |
| `amount` | `1 - 1,000,000` | `10000` |

<br />

## Project Structure

```
saudi-stock-ai-analyzer/
│
├── backend/
│   ├── api/routers/         # Route handlers
│   ├── backtest/            # Backtesting engine
│   ├── data/                # Data loaders
│   ├── database/            # Position storage
│   ├── models/              # ML models
│   ├── strategy/            # Trading logic
│   ├── utils/               # Validators, config
│   ├── app.py               # FastAPI app
│   └── main.py              # Core analyzer
│
├── frontend/
│   ├── src/
│   │   ├── api/             # API client
│   │   ├── components/      # UI components
│   │   ├── hooks/           # React hooks
│   │   ├── pages/           # Route pages
│   │   └── styles/          # CSS
│   └── package.json
│
└── README.md
```

<br />

## Market Coverage

The platform covers **200+ stocks** across all Tadawul sectors:

| Sector | Examples |
|:-------|:---------|
| Energy | Saudi Aramco (2222), Petro Rabigh, Bahri |
| Banks | Al Rajhi (1120), SNB (1182), Alinma (1180) |
| Materials | SABIC (2010), Ma'aden (1211), Saudi Kayan |
| Telecom | STC (7010), Mobily (7020), Zain (7030) |
| Healthcare | Dr. Sulaiman Al Habib (4013), Mouwasat |
| Utilities | Saudi Electricity (5110), ACWA Power |
| Consumer | Almarai (2280), Jarir (4190), Al Othaim |
| REITs | Riyad REIT (4330), Jadwa REIT (4331) |

<br />

## Testing

```bash
# Backend
cd backend && pytest

# Frontend
cd frontend && npm test
```

<br />

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

<br />

---

<div align="center">
<sub>Built by <a href="https://github.com/Abdulrahman-S-Asiri">Abdulrahman Asiri</a></sub>
</div>
