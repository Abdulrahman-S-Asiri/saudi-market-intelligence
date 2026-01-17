# Saudi Stock AI Analyzer

A full-stack AI-powered stock analysis platform for the Saudi Arabian market (Tadawul). Combines deep learning models with technical analysis to generate trading signals and insights.

## Features

- **AI-Powered Predictions**: Ensemble model combining LSTM neural networks with XGBoost for accurate price direction forecasting
- **Technical Analysis**: Comprehensive indicators including RSI, MACD, Bollinger Bands, ADX, and more
- **Trading Signals**: Automated BUY/SELL/HOLD recommendations with confidence scores
- **Risk Management**: Stop-loss and take-profit calculations based on ATR
- **Backtesting Engine**: Historical performance evaluation with detailed metrics
- **Real-Time Dashboard**: Modern React frontend with interactive charts
- **REST API**: FastAPI backend with comprehensive endpoints

## Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI** - Modern async web framework
- **PyTorch** - Deep learning models (LSTM with attention)
- **scikit-learn** - Data preprocessing and metrics
- **XGBoost** - Gradient boosting for ensemble predictions
- **yfinance** - Market data from Yahoo Finance
- **pandas/numpy** - Data manipulation

### Frontend
- **React 18** - UI framework
- **Recharts** - Charting library
- **Lightweight Charts** - TradingView-style candlestick charts
- **Framer Motion** - Animations
- **Axios** - HTTP client

## Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- pip and npm

### Backend Setup

```bash
cd saudi-stock-ai-analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

## Usage

### Running the Backend

```bash
# From saudi-stock-ai-analyzer directory
python app.py
```

The API will be available at `http://localhost:8000`

### Running the Frontend

```bash
# From frontend directory
npm start
```

The dashboard will open at `http://localhost:3000`

### Running Both (Development)

Terminal 1:
```bash
cd saudi-stock-ai-analyzer && python app.py
```

Terminal 2:
```bash
cd saudi-stock-ai-analyzer/frontend && npm start
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stocks` | GET | List available Saudi stocks |
| `/api/analyze/{symbol}` | GET | Full stock analysis with ML prediction |
| `/api/chart/{symbol}` | GET | OHLCV data with technical indicators |
| `/api/predict/{symbol}` | GET | ML prediction only |
| `/api/backtest/{symbol}` | GET | Historical backtest results |
| `/api/signals/history/{symbol}` | GET | Past trading signals |
| `/api/compare` | GET | Compare multiple stocks |
| `/api/risk/{symbol}` | GET | Risk metrics (VaR, Sharpe, etc.) |
| `/api/health` | GET | API health check |

### Example API Call

```bash
curl http://localhost:8000/api/analyze/2222?period=6mo
```

## Project Structure

```
saudi-stock-ai-analyzer/
├── app.py                 # FastAPI server
├── main.py                # Analysis orchestrator
├── requirements.txt       # Python dependencies
│
├── data/
│   ├── data_loader.py     # Yahoo Finance data fetching
│   ├── data_collector.py  # Data collection utilities
│   └── data_preprocessor.py # Feature engineering
│
├── models/
│   ├── lstm_model.py      # LSTM with attention mechanism
│   ├── transformer_model.py # Transformer architecture
│   └── ensemble_model.py  # LSTM + XGBoost ensemble
│
├── strategy/
│   ├── trading_strategy.py # Signal generation logic
│   └── risk_manager.py    # Position sizing & risk
│
├── backtest/
│   ├── backtest_engine.py # Historical simulation
│   └── performance_metrics.py # Sharpe, drawdown, etc.
│
├── utils/
│   ├── config.py          # Configuration & stock list
│   └── logger.py          # Logging utilities
│
└── frontend/
    ├── package.json
    ├── public/
    └── src/
        ├── App.jsx        # Main application
        ├── components/    # React components
        ├── hooks/         # Custom hooks
        └── styles/        # CSS styles
```

## Supported Stocks

The analyzer supports major Saudi stocks including:

| Symbol | Company | Sector |
|--------|---------|--------|
| 2222 | Saudi Aramco | Energy |
| 1120 | Al Rajhi Bank | Banking |
| 2010 | SABIC | Materials |
| 1180 | Al Ahli Bank | Banking |
| 2380 | Petrochemical | Materials |

See `utils/config.py` for the complete list.

## Model Architecture

### LSTM with Attention
- Bidirectional LSTM layers
- Multi-head attention mechanism
- Dropout regularization
- Trained on 20+ technical features

### Ensemble Model
- Combines LSTM neural network with XGBoost
- Dynamic weight adjustment based on validation performance
- Model caching for faster subsequent predictions

## Risk Metrics

The platform calculates comprehensive risk metrics:

- **Volatility** - Annualized standard deviation
- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside risk-adjusted returns
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Value at Risk (VaR)** - 95% and 99% confidence levels
- **Expected Shortfall (CVaR)** - Average loss beyond VaR

## Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Trading stocks involves significant risk of loss. Always:

- Do your own research
- Consult with a licensed financial advisor
- Never invest more than you can afford to lose
- Past performance does not guarantee future results

The developers are not responsible for any financial losses incurred from using this software.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
