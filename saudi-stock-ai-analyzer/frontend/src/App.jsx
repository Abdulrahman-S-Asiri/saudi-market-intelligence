/**
 * Saudi Stock AI Analyzer - 2026 Edition
 * Clean, Modern, Minimalist Design
 * BiLSTM + Multi-Head Attention Model
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './styles/App.css';
import PositionManager from './components/PositionManager';
import LandingPage from './components/LandingPage';
import MarketScanner from './components/MarketScanner';
import usePositions from './hooks/usePositions';

const API_BASE = 'http://localhost:8000';

// Signal Badge Component
const SignalBadge = ({ signal, confidence }) => {
  const getSignalClass = () => {
    if (!signal) return 'neutral';
    const s = signal.toLowerCase();
    if (s.includes('buy') || s.includes('strong buy')) return 'buy';
    if (s.includes('sell')) return 'sell';
    return 'hold';
  };

  return (
    <div className={`signal-badge ${getSignalClass()}`}>
      <span className="signal-icon">
        {getSignalClass() === 'buy' ? '↑' : getSignalClass() === 'sell' ? '↓' : '→'}
      </span>
      <span className="signal-text">{signal || 'ANALYZING'}</span>
      {confidence && <span className="signal-confidence">{confidence}%</span>}
    </div>
  );
};

// Indicator Card Component
const IndicatorCard = ({ label, value, status }) => (
  <div className={`indicator-card ${status || ''}`}>
    <span className="indicator-label">{label}</span>
    <span className="indicator-value">{value ?? '-'}</span>
  </div>
);

// Mini Chart Component (SVG Sparkline)
const MiniChart = ({ data }) => {
  if (!data || data.length === 0) return <div className="mini-chart empty">No data</div>;

  const prices = data.slice(-60).map(d => d.close);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;

  const points = prices.map((price, i) => {
    const x = (i / (prices.length - 1)) * 100;
    const y = 100 - ((price - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  const isUp = prices[prices.length - 1] >= prices[0];

  return (
    <div className="mini-chart">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <linearGradient id="chartGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={isUp ? "#00ffa3" : "#ff4757"} stopOpacity="0.4"/>
            <stop offset="100%" stopColor={isUp ? "#00ffa3" : "#ff4757"} stopOpacity="0"/>
          </linearGradient>
        </defs>
        <polygon points={`0,100 ${points} 100,100`} fill="url(#chartGrad)" />
        <polyline
          points={points}
          fill="none"
          stroke={isUp ? "#00ffa3" : "#ff4757"}
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </div>
  );
};

// TASI Sector Configuration with icons
const TASI_SECTORS = [
  { id: 'all', name: 'All Sectors', icon: '📊' },
  { id: 'Energy', name: 'Energy', icon: '⚡' },
  { id: 'Materials', name: 'Materials', icon: '🏗️' },
  { id: 'Banks', name: 'Banks', icon: '🏦' },
  { id: 'Diversified Financials', name: 'Financials', icon: '💰' },
  { id: 'Insurance', name: 'Insurance', icon: '🛡️' },
  { id: 'Telecommunication Services', name: 'Telecom', icon: '📡' },
  { id: 'Utilities', name: 'Utilities', icon: '💡' },
  { id: 'Real Estate', name: 'Real Estate', icon: '🏢' },
  { id: 'REITs', name: 'REITs', icon: '🏠' },
  { id: 'Consumer Services', name: 'Consumer', icon: '🛒' },
  { id: 'Food & Beverages', name: 'Food & Bev', icon: '🍽️' },
  { id: 'Health Care', name: 'Healthcare', icon: '🏥' },
  { id: 'Pharma', name: 'Pharma', icon: '💊' },
  { id: 'Capital Goods', name: 'Industrial', icon: '🏭' },
  { id: 'Transportation', name: 'Transport', icon: '🚚' },
  { id: 'Software & Services', name: 'Tech', icon: '💻' },
  { id: 'Consumer Durables', name: 'Durables', icon: '📦' },
];

// Sector Filter Chips Component
const SectorChips = ({ sectors, selectedSector, onSelect }) => {
  return (
    <div className="sector-chips-container">
      <div className="sector-chips">
        {sectors.map(sector => (
          <motion.button
            key={sector.id}
            className={`sector-chip ${selectedSector === sector.id ? 'active' : ''}`}
            onClick={() => onSelect(sector.id)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="chip-icon">{sector.icon}</span>
            <span className="chip-name">{sector.name}</span>
          </motion.button>
        ))}
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [stocks, setStocks] = useState([
    { symbol: '2222', name: 'Saudi Aramco', sector: 'Energy' },
    { symbol: '1120', name: 'Al Rajhi Bank', sector: 'Banks' },
    { symbol: '2010', name: 'SABIC', sector: 'Materials' },
    { symbol: '7010', name: 'STC', sector: 'Telecommunication Services' },
  ]);
  const [selectedStock, setSelectedStock] = useState('2222');
  const [selectedSector, setSelectedSector] = useState('all');
  const [analysis, setAnalysis] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('analysis'); // 'analysis', 'signals', or 'positions'

  // Handle entering the main app from landing page
  const handleEnterApp = useCallback(() => {
    setShowLanding(false);
  }, []);

  // Filter stocks by selected sector
  const filteredStocks = useMemo(() => {
    if (selectedSector === 'all') return stocks;
    return stocks.filter(stock => stock.sector === selectedSector);
  }, [stocks, selectedSector]);

  // Handle sector change
  const handleSectorChange = useCallback((sectorId) => {
    setSelectedSector(sectorId);
    // If current stock is not in the new sector, select the first stock of that sector
    if (sectorId !== 'all') {
      const sectorStocks = stocks.filter(s => s.sector === sectorId);
      if (sectorStocks.length > 0 && !sectorStocks.find(s => s.symbol === selectedStock)) {
        setSelectedStock(sectorStocks[0].symbol);
      }
    }
  }, [stocks, selectedStock]);

  // Position manager hook
  const { createFromSignal, fetchPositions } = usePositions(false);

  // Handle creating position from signal
  const handleCreatePosition = useCallback(async (symbol, price) => {
    try {
      await createFromSignal(symbol, price);
      // Show success notification (could be improved with a toast)
      alert(`Position created for ${symbol} at ${price} SAR`);
      // Switch to positions tab
      setActiveTab('positions');
    } catch (err) {
      alert('Failed to create position: ' + err.message);
    }
  }, [createFromSignal]);

  // Fetch stocks list
  useEffect(() => {
    fetch(`${API_BASE}/api/stocks`)
      .then(res => res.json())
      .then(data => { if (data.stocks) setStocks(data.stocks); })
      .catch(() => {});
  }, []);

  // Fetch analysis when stock changes
  useEffect(() => {
    if (!selectedStock) return;
    setLoading(true);
    setError(null);

    Promise.all([
      fetch(`${API_BASE}/api/analyze/${selectedStock}?period=6mo&train_model=true`),
      fetch(`${API_BASE}/api/chart/${selectedStock}?period=6mo`)
    ])
      .then(async ([analysisRes, chartRes]) => {
        const analysisData = await analysisRes.json();
        const chartDataRes = await chartRes.json();
        if (analysisData.error) throw new Error(analysisData.error);
        setAnalysis(analysisData);
        setChartData(chartDataRes.data || []);
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [selectedStock]);

  const currentStock = stocks.find(s => s.symbol === selectedStock) || {};
  const priceChange = analysis?.performance?.total_return || 0;

  // Show Landing Page
  if (showLanding) {
    return <LandingPage onEnter={handleEnterApp} />;
  }

  return (
    <motion.div
      className="app"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
    >
      {/* Background */}
      <div className="bg-gradient" />
      <div className="bg-grid" />

      {/* Header */}
      <header className="header">
        <div className="header-top">
          <div className="logo">
            <span className="logo-icon">📊</span>
            <div>
              <h1>TASI AI</h1>
              <span className="logo-sub">BiLSTM + Attention v3.0</span>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="nav-tabs">
            <button
              className={`nav-tab ${activeTab === 'analysis' ? 'active' : ''}`}
              onClick={() => setActiveTab('analysis')}
            >
              Analysis
            </button>
            <button
              className={`nav-tab ${activeTab === 'signals' ? 'active' : ''}`}
              onClick={() => setActiveTab('signals')}
            >
              Signals
            </button>
            <button
              className={`nav-tab ${activeTab === 'positions' ? 'active' : ''}`}
              onClick={() => setActiveTab('positions')}
            >
              Positions
            </button>
          </div>

          <div className="status">
            <span className={`dot ${loading ? 'loading' : 'ready'}`} />
            {loading ? 'Analyzing...' : 'Ready'}
          </div>
        </div>

        {/* Sector Filter & Stock Selector */}
        {activeTab === 'analysis' && (
          <div className="stock-selector-section">
            {/* Sector Chips */}
            <SectorChips
              sectors={TASI_SECTORS}
              selectedSector={selectedSector}
              onSelect={handleSectorChange}
            />

            {/* Stock Dropdown */}
            <div className="stock-dropdown-wrapper">
              <select
                className="stock-select"
                value={selectedStock}
                onChange={(e) => setSelectedStock(e.target.value)}
              >
                {filteredStocks.map(stock => (
                  <option key={stock.symbol} value={stock.symbol}>
                    {stock.symbol} - {stock.name}
                  </option>
                ))}
              </select>
              <span className="stock-count">
                {filteredStocks.length} stocks
              </span>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="main">
        {/* Positions Tab */}
        {activeTab === 'positions' && (
          <PositionManager stocks={stocks} />
        )}

        {/* Signals Tab */}
        {activeTab === 'signals' && (
          <MarketScanner
            onStockSelect={(symbol) => {
              setSelectedStock(symbol);
              setActiveTab('analysis');
            }}
          />
        )}

        {/* Analysis Tab */}
        {activeTab === 'analysis' && error && <div className="error">⚠️ {error}</div>}

        {activeTab === 'analysis' && loading && !analysis && (
          <div className="loading-screen">
            <div className="spinner" />
            <p>Training AI Model...</p>
            <span>BiLSTM + Multi-Head Attention</span>
          </div>
        )}

        {activeTab === 'analysis' && analysis && (
          <>
            {/* Hero Card */}
            <section className="hero glass">
              <div className="hero-info">
                <h2>{currentStock.name}</h2>
                <div className="hero-meta">
                  <span className="symbol">{selectedStock}.SR</span>
                  <span className="sector">{currentStock.sector}</span>
                </div>
                <div className="price">
                  <span className="price-value">{analysis.signal?.price?.toFixed(2)}</span>
                  <span className="price-currency">SAR</span>
                  <span className={`price-change ${priceChange >= 0 ? 'up' : 'down'}`}>
                    {priceChange >= 0 ? '+' : ''}{priceChange?.toFixed(2)}%
                  </span>
                </div>
              </div>

              <div className="hero-chart">
                <MiniChart data={chartData} />
              </div>

              <div className="hero-signal">
                <SignalBadge
                  signal={analysis.signal?.action}
                  confidence={analysis.signal?.confidence}
                />
                {/* Open Position Button for BUY signals */}
                {analysis.signal?.action === 'BUY' && (
                  <button
                    className="btn-open-from-signal"
                    onClick={() => handleCreatePosition(selectedStock, analysis.signal.price)}
                  >
                    + Open Position
                  </button>
                )}
              </div>
            </section>

            {/* Stats Grid */}
            <section className="stats-grid">
              <IndicatorCard
                label="RSI"
                value={analysis.indicators?.rsi?.toFixed(1)}
                status={analysis.indicators?.rsi < 30 ? 'oversold' : analysis.indicators?.rsi > 70 ? 'overbought' : ''}
              />
              <IndicatorCard
                label="MACD"
                value={analysis.indicators?.macd?.toFixed(4)}
                status={analysis.indicators?.macd > 0 ? 'positive' : 'negative'}
              />
              <IndicatorCard
                label="Volatility"
                value={`${analysis.performance?.volatility?.toFixed(1)}%`}
              />
              <IndicatorCard
                label="Sharpe"
                value={analysis.performance?.sharpe_ratio?.toFixed(2)}
                status={analysis.performance?.sharpe_ratio > 1 ? 'positive' : ''}
              />
              <IndicatorCard
                label="Win Rate"
                value={`${analysis.performance?.win_rate?.toFixed(0)}%`}
                status={analysis.performance?.win_rate > 50 ? 'positive' : ''}
              />
              <IndicatorCard
                label="Max DD"
                value={`${analysis.performance?.max_drawdown?.toFixed(1)}%`}
                status="negative"
              />
            </section>

            {/* AI Prediction */}
            {analysis.ml_prediction && (
              <section className="ai-section glass">
                <div className="ai-header">
                  <span>🤖</span>
                  <h3>AI Prediction</h3>
                  <span className="badge">BiLSTM + Attention</span>
                </div>
                <div className="ai-content">
                  <div className={`direction ${analysis.ml_prediction.direction?.toLowerCase()}`}>
                    {analysis.ml_prediction.direction === 'UP' ? '📈' : analysis.ml_prediction.direction === 'DOWN' ? '📉' : '➡️'}
                    <span>{analysis.ml_prediction.direction}</span>
                  </div>
                  <div className="confidence">
                    <span className="conf-value">{analysis.ml_prediction.confidence?.toFixed(1)}%</span>
                    <span className="conf-label">Confidence</span>
                  </div>
                </div>
                {analysis.signal?.reasons && (
                  <ul className="reasons">
                    {analysis.signal.reasons.slice(0, 3).map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                )}
              </section>
            )}

            {/* Risk Metrics */}
            <section className="risk-section glass">
              <h3>📊 Risk Metrics</h3>
              <div className="risk-grid">
                <div className="risk-item">
                  <span className="risk-label">VaR 95%</span>
                  <span className="risk-value">{analysis.risk?.var_95?.toFixed(2)}%</span>
                </div>
                <div className="risk-item">
                  <span className="risk-label">Sortino</span>
                  <span className="risk-value">{analysis.risk?.sortino_ratio?.toFixed(2)}</span>
                </div>
                <div className="risk-item">
                  <span className="risk-label">Calmar</span>
                  <span className="risk-value">{analysis.risk?.calmar_ratio?.toFixed(2)}</span>
                </div>
                <div className="risk-item">
                  <span className="risk-label">Regime</span>
                  <span className="risk-value regime">{analysis.signal?.market_regime}</span>
                </div>
              </div>
            </section>

            {/* Market Scanner - Actionable Insights */}
            <section className="market-scanner-section">
              <MarketScanner
                onStockSelect={(symbol) => {
                  setSelectedStock(symbol);
                  // Scroll to top to see the new analysis
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
              />
            </section>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>TASI AI v3.0 • BiLSTM + Multi-Head Attention • 2026</p>
      </footer>
    </motion.div>
  );
}

export default App;
