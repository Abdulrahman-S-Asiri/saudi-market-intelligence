import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import StockSelector from './components/StockSelector';
import TimeframeSelector from './components/TimeframeSelector';
import PriceChart from './components/PriceChart';
import RSIChart from './components/RSIChart';
import MACDChart from './components/MACDChart';
import SignalPanel from './components/SignalPanel';
import SignalHistory from './components/SignalHistory';
import MetricsCard from './components/MetricsCard';
import IndicatorsTable from './components/IndicatorsTable';
import useStockData from './hooks/useStockData';
import './styles/App.css';

function App() {
  const [selectedStock, setSelectedStock] = useState('2222');
  const [timeframe, setTimeframe] = useState('6mo');
  const [activeTab, setActiveTab] = useState('overview');
  const [showNotification, setShowNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState('');

  const {
    analysis,
    chartData,
    stocks,
    signalHistory,
    loading,
    error,
    fetchAnalysis,
    fetchChartData,
    fetchSignalHistory
  } = useStockData();

  const showToast = useCallback((message) => {
    setNotificationMessage(message);
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 3000);
  }, []);

  useEffect(() => {
    if (selectedStock) {
      fetchAnalysis(selectedStock, timeframe);
      fetchChartData(selectedStock, timeframe);
    }
  }, [selectedStock, timeframe, fetchAnalysis, fetchChartData]);

  const handleStockChange = (symbol) => {
    setSelectedStock(symbol);
    showToast(`Switched to ${symbol}`);
  };

  const handleTimeframeChange = (tf) => {
    setTimeframe(tf);
    showToast(`Timeframe changed to ${tf}`);
  };

  const handleRefresh = () => {
    if (selectedStock) {
      fetchAnalysis(selectedStock, timeframe);
      fetchChartData(selectedStock, timeframe);
      showToast('Data refreshed');
    }
  };

  return (
    <div className="app">
      {/* Animated Background */}
      <div className="background-effects">
        <div className="bg-gradient"></div>
        <div className="bg-grid"></div>
      </div>

      {/* Toast Notification */}
      <AnimatePresence>
        {showNotification && (
          <motion.div
            className="toast-notification"
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -50 }}
          >
            {notificationMessage}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <header className="header glass-card">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 3v18h18" />
                <path d="M18 9l-5-6-4 10-3-3" />
              </svg>
            </div>
            <div className="logo-text">
              <h1>Saudi Stock AI Analyzer</h1>
              <p>AI-powered analysis for Tadawul market</p>
            </div>
          </div>
          <div className="header-actions">
            <button className="refresh-btn" onClick={handleRefresh} disabled={loading}>
              <motion.span
                animate={loading ? { rotate: 360 } : {}}
                transition={{ duration: 1, repeat: loading ? Infinity : 0, ease: 'linear' }}
              >
                {/* Refresh icon */}
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M1 4v6h6M23 20v-6h-6" />
                  <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
                </svg>
              </motion.span>
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Controls Section */}
        <motion.div
          className="controls-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <StockSelector
            stocks={stocks}
            selectedStock={selectedStock}
            onSelect={handleStockChange}
          />
          <TimeframeSelector
            selected={timeframe}
            onSelect={handleTimeframeChange}
          />
        </motion.div>

        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div
              className="error-message glass-card"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              <span className="error-icon">!</span>
              <p>Error: {error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading State */}
        <AnimatePresence>
          {loading && !analysis && (
            <motion.div
              className="loading-container"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="loading-card glass-card">
                <div className="loading-spinner large"></div>
                <h3>Analyzing Stock Data</h3>
                <p>Training AI model and calculating indicators...</p>
                <div className="loading-steps">
                  <div className="step active">Fetching data</div>
                  <div className="step">Processing</div>
                  <div className="step">AI Analysis</div>
                  <div className="step">Generating signals</div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Dashboard */}
        {analysis && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            {/* Stock Header */}
            <motion.div
              className="stock-header glass-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="stock-info">
                <h2>{analysis.name}</h2>
                <span className="symbol-badge">{analysis.symbol}</span>
                <span className="sector-badge">{analysis.sector}</span>
              </div>
              <div className="stock-price">
                <span className="price-value">{analysis.signal?.price?.toFixed(2)}</span>
                <span className="price-currency">SAR</span>
                {analysis.trend && (
                  <span
                    className={`price-change ${
                      analysis.trend.change_5d >= 0 ? 'positive' : 'negative'
                    }`}
                  >
                    {analysis.trend.change_5d >= 0 ? '+' : ''}
                    {analysis.trend.change_5d}%
                  </span>
                )}
              </div>
            </motion.div>

            {/* Navigation Tabs */}
            <div className="tab-navigation">
              {['overview', 'charts', 'analysis'].map((tab) => (
                <button
                  key={tab}
                  className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab)}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  {activeTab === tab && (
                    <motion.div
                      className="tab-indicator"
                      layoutId="tabIndicator"
                    />
                  )}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <AnimatePresence mode="wait">
              {activeTab === 'overview' && (
                <motion.div
                  key="overview"
                  className="dashboard-grid overview-grid"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="signal-section">
                    <SignalPanel
                      signal={analysis.signal}
                      lstmPrediction={analysis.lstm_prediction}
                      loading={loading}
                    />
                  </div>

                  <div className="chart-section">
                    <PriceChart
                      data={chartData?.data || []}
                      loading={loading}
                      timeframe={timeframe}
                    />
                  </div>

                  <div className="metrics-section">
                    <MetricsCard
                      performance={analysis.performance}
                      periodReturns={analysis.period_returns}
                      trend={analysis.trend}
                    />
                  </div>

                  <div className="indicators-section">
                    <IndicatorsTable
                      indicators={analysis.indicators}
                      supportResistance={analysis.support_resistance}
                    />
                  </div>
                </motion.div>
              )}

              {activeTab === 'charts' && (
                <motion.div
                  key="charts"
                  className="dashboard-grid charts-grid"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="main-chart-section">
                    <PriceChart
                      data={chartData?.data || []}
                      loading={loading}
                      timeframe={timeframe}
                    />
                  </div>

                  <div className="indicator-charts">
                    <RSIChart data={chartData?.data || []} loading={loading} />
                    <MACDChart data={chartData?.data || []} loading={loading} />
                  </div>
                </motion.div>
              )}

              {activeTab === 'analysis' && (
                <motion.div
                  key="analysis"
                  className="dashboard-grid analysis-grid"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="signal-detail-section">
                    <SignalPanel
                      signal={analysis.signal}
                      lstmPrediction={analysis.lstm_prediction}
                      loading={loading}
                    />
                  </div>

                  <div className="history-section">
                    <SignalHistory signals={signalHistory} loading={loading} />
                  </div>

                  <div className="full-metrics-section">
                    <MetricsCard
                      performance={analysis.performance}
                      periodReturns={analysis.period_returns}
                      trend={analysis.trend}
                    />
                    <IndicatorsTable
                      indicators={analysis.indicators}
                      supportResistance={analysis.support_resistance}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer glass-card">
        <div className="footer-content">
          <p>Saudi Stock AI Analyzer v2.0 - Professional Edition</p>
          <p>Data provided by Yahoo Finance. For educational purposes only.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
