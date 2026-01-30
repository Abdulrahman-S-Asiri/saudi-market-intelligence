<div align="center">

# 🇸🇦 TASI AI Analyzer

### Institutional-Grade Analysis for the Saudi Stock Market

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stack: FastAPI/React](https://img.shields.io/badge/Stack-FastAPI%20%7C%20React-blue.svg)](#tech-stack)
[![Methodology: Vibe Coding](https://img.shields.io/badge/Methodology-Vibe%20Coding-purple.svg)](#-built-via-vibe-coding)

<br />

**Real-time AI predictions • Uncertainty quantification • Market regime detection**

[Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [License](#-license)

</div>

---

## 🎯 Built via Vibe Coding

> **This project demonstrates the power of Vibe Coding** — a methodology that leverages high-bandwidth human-AI collaboration to engineer complex, production-ready systems rapidly.

Rather than traditional line-by-line coding, Vibe Coding focuses on:

- 🧠 **Intent-driven development** — Describe what you want, not how to build it
- ⚡ **Rapid iteration** — From concept to production in hours, not weeks
- 🔄 **Continuous refinement** — Human creativity + AI execution in tight feedback loops
- 🏗️ **Full-stack delivery** — Complete systems, not just snippets

This entire platform — backend, frontend, ML models, and infrastructure — was engineered through Vibe Coding, showcasing what's possible when humans and AI collaborate at maximum bandwidth.

---

## ✨ Key Features

### 🤖 **Real-time LSTM & Transformer Predictions**
Advanced BiLSTM architecture with Multi-Head Attention mechanisms, trained on 35+ technical indicators for high-accuracy directional forecasting.

### 📊 **Monte Carlo Dropout for True Uncertainty Estimation**
Go beyond point predictions. Our MC Dropout implementation provides calibrated confidence intervals, so you know *how certain* the model is about each prediction.

### 📈 **Market Regime Detection (Bull/Bear/Sideways)**
Hidden Markov Model-based regime classification using the TASI Index. Automatically adapts trading strategies to current market conditions.

### 🎨 **Interactive React Dashboard**
Beautiful, responsive UI with real-time candlestick charts, technical indicators, signal history, and position management — all powered by TradingView's Lightweight Charts.

### 🛡️ **Comprehensive Risk Metrics**
Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Value at Risk (VaR), Expected Shortfall — institutional-grade risk analytics at your fingertips.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Python 3.10+, FastAPI, PyTorch, scikit-learn |
| **Frontend** | React 18, Lightweight Charts, Framer Motion |
| **ML Models** | BiLSTM, Multi-Head Attention, HMM, XGBoost |
| **Data** | yfinance, pandas, numpy |

---

## 📦 Installation

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Abdulrahman-S-Asiri/saudi-market-intelligence.git
cd saudi-market-intelligence/saudi-stock-ai-analyzer

# Start all services
docker-compose up
```

The application will be available at:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### Option 2: Manual Installation

**Backend Setup:**
```bash
cd saudi-stock-ai-analyzer/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

**Frontend Setup:**
```bash
cd saudi-stock-ai-analyzer/frontend

# Install dependencies
npm install

# Start the development server
npm start
```

### Quick Start (Windows)

Simply double-click `start.bat` to launch both servers automatically.

---

## 🚀 Usage

1. **Select a Stock** — Choose from 200+ TASI-listed stocks across 21 sectors
2. **View Analysis** — Real-time AI predictions with confidence scores
3. **Check Signals** — BUY/SELL/HOLD recommendations with technical justification
4. **Monitor Regime** — See current market conditions (Bull/Bear/Sideways)
5. **Track Positions** — Manage your portfolio with built-in position tracking

---

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/stocks` | List all available stocks |
| `GET /api/analyze/{symbol}` | Full analysis with ML prediction |
| `GET /api/chart/{symbol}` | OHLCV data with indicators |
| `GET /api/predict/{symbol}` | ML prediction with uncertainty |
| `GET /api/regime` | Current market regime |
| `GET /api/scanner` | Top gainers/losers scan |

**Example:**
```bash
curl http://localhost:8000/api/analyze/2222?period=6mo
```

---

## 📁 Project Structure

```
saudi-stock-ai-analyzer/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── main.py             # Analysis orchestrator
│   ├── models/             # LSTM, Attention, HMM
│   ├── data/               # Data loading & preprocessing
│   ├── strategy/           # Trading signals & risk
│   ├── backtest/           # Historical validation
│   └── utils/              # Config & utilities
│
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── hooks/          # Custom hooks
│   │   └── styles/         # CSS styles
│   └── public/
│
├── start.bat               # Quick launcher
└── README.md
```

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice. Trading involves significant risk of loss. Always conduct your own research and consult with licensed financial advisors.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](../LICENSE) file for details.

---

<div align="center">

**© 2026 Abdulrahman Asiri. All rights reserved.**

*Engineered via Vibe Coding* 🚀

</div>
