/**
 * Market Scanner Component
 * Displays Top Projected Gainers and Losers based on AI predictions
 * Includes Market Regime Detection (HMM-based Bull/Bear/Sideways)
 * Provides actionable insights at a glance
 */

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './MarketScanner.css';

const API_BASE = 'http://localhost:8000';

// Market Regime Badge Component
const MarketRegimeBadge = ({ regime }) => {
  if (!regime) return null;

  const regimeStyles = {
    bull: { bg: 'rgba(0, 255, 163, 0.15)', border: 'rgba(0, 255, 163, 0.4)', color: '#00ffa3' },
    bear: { bg: 'rgba(255, 71, 87, 0.15)', border: 'rgba(255, 71, 87, 0.4)', color: '#ff4757' },
    sideways: { bg: 'rgba(255, 193, 7, 0.15)', border: 'rgba(255, 193, 7, 0.4)', color: '#ffc107' },
  };

  const style = regimeStyles[regime.regime] || regimeStyles.sideways;

  return (
    <motion.div
      className="regime-badge"
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      style={{
        background: style.bg,
        border: `1px solid ${style.border}`,
        color: style.color,
        padding: '4px 10px',
        borderRadius: '12px',
        fontSize: '12px',
        fontWeight: '600',
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
      }}
      title={`${regime.regime_name} - ${regime.confidence}% confidence`}
    >
      <span>{regime.emoji}</span>
      <span>{regime.regime_name}</span>
    </motion.div>
  );
};

// Market Warning Banner
const MarketWarningBanner = ({ warning, regime }) => {
  if (!warning) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="market-warning-banner"
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
        style={{
          background: regime === 'bear'
            ? 'linear-gradient(90deg, rgba(255, 71, 87, 0.15), rgba(255, 71, 87, 0.05))'
            : 'linear-gradient(90deg, rgba(255, 193, 7, 0.15), rgba(255, 193, 7, 0.05))',
          borderLeft: regime === 'bear'
            ? '3px solid #ff4757'
            : '3px solid #ffc107',
          padding: '10px 16px',
          fontSize: '13px',
          color: regime === 'bear' ? '#ff6b7a' : '#ffc107',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        <span style={{ fontSize: '16px' }}>
          {regime === 'bear' ? '⚠️' : '⚡'}
        </span>
        <span>{warning}</span>
      </motion.div>
    </AnimatePresence>
  );
};

// Stock Item Row
const StockItem = ({ stock, type, rank, onClick }) => {
  const isGainer = type === 'gainer';
  const changeValue = stock.predicted_change || 0;

  return (
    <motion.div
      className={`scanner-item ${type}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: rank * 0.08 }}
      whileHover={{ scale: 1.02, x: isGainer ? 4 : -4 }}
      whileTap={{ scale: 0.98 }}
      onClick={() => onClick(stock.symbol)}
    >
      <div className="scanner-item-rank">{rank + 1}</div>

      <div className="scanner-item-info">
        <div className="scanner-item-name">{stock.name}</div>
        <div className="scanner-item-symbol">{stock.symbol}.SR</div>
      </div>

      <div className="scanner-item-price">
        <span className="price-value">{stock.price?.toFixed(2)}</span>
        <span className="price-label">SAR</span>
      </div>

      <div className={`scanner-item-change ${isGainer ? 'positive' : 'negative'}`}>
        <span className="change-icon">{isGainer ? '▲' : '▼'}</span>
        <span className="change-value">
          {isGainer ? '+' : ''}{changeValue.toFixed(2)}%
        </span>
      </div>

      <div className="scanner-item-confidence">
        <div className="confidence-bar">
          <motion.div
            className="confidence-fill"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(stock.ml_confidence || 0, 100)}%` }}
            transition={{ delay: rank * 0.08 + 0.2, duration: 0.4 }}
          />
        </div>
        <span className="confidence-text">{stock.ml_confidence?.toFixed(0)}%</span>
      </div>
    </motion.div>
  );
};

// Loading Placeholder
const LoadingRows = ({ count = 5 }) => (
  <>
    {[...Array(count)].map((_, i) => (
      <div key={i} className="scanner-item-skeleton">
        <div className="skeleton-rank" />
        <div className="skeleton-info">
          <div className="skeleton-name" />
          <div className="skeleton-symbol" />
        </div>
        <div className="skeleton-price" />
        <div className="skeleton-change" />
        <div className="skeleton-confidence" />
      </div>
    ))}
  </>
);

// Main Market Scanner Component
const MarketScanner = ({ onStockSelect, compact = false }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [marketStatus, setMarketStatus] = useState(null);

  // Fetch market status (regime detection)
  const fetchMarketStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/market-status`);
      if (response.ok) {
        const result = await response.json();
        setMarketStatus(result);
      }
    } catch (err) {
      console.error('Failed to fetch market status:', err);
    }
  }, []);

  // Fetch market rankings
  const fetchData = useCallback(async (forceRefresh = false) => {
    try {
      setError(null);
      if (forceRefresh) {
        setRefreshing(true);
      } else if (!data) {
        setLoading(true);
      }

      const response = await fetch(
        `${API_BASE}/api/market-rankings?force_refresh=${forceRefresh}&quick_scan=true`
      );

      if (!response.ok) throw new Error('Failed to fetch market data');

      const result = await response.json();

      if (result.status === 'updating') {
        setError('Scanning market... Please wait.');
        return;
      }

      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [data]);

  useEffect(() => {
    fetchData();
    fetchMarketStatus();
  }, []);

  const handleStockClick = (symbol) => {
    if (onStockSelect) {
      onStockSelect(symbol);
    }
  };

  const handleRefresh = () => {
    fetchData(true);
  };

  // Format time ago
  const getTimeAgo = (minutes) => {
    if (!minutes) return 'now';
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${Math.round(minutes)}m ago`;
    return `${Math.round(minutes / 60)}h ago`;
  };

  return (
    <div className={`market-scanner ${compact ? 'compact' : ''}`}>
      {/* Header */}
      <div className="scanner-header">
        <div className="scanner-title">
          <span className="scanner-icon">📡</span>
          <h3>Market Scanner</h3>
          {/* Market Regime Badge */}
          <MarketRegimeBadge regime={marketStatus} />
        </div>
        <div className="scanner-actions">
          {data?.cache_info && (
            <span className="scanner-updated">
              {getTimeAgo(data.cache_info.cache_age_minutes)}
            </span>
          )}
          <motion.button
            className="scanner-refresh"
            onClick={handleRefresh}
            disabled={refreshing || loading}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            title="Refresh data"
          >
            <span className={refreshing ? 'spinning' : ''}>↻</span>
          </motion.button>
        </div>
      </div>

      {/* Market Warning Banner (shows when bearish) */}
      <MarketWarningBanner
        warning={marketStatus?.warning}
        regime={marketStatus?.regime}
      />

      {/* Error State */}
      {error && <div className="scanner-error">{error}</div>}

      {/* Content */}
      <div className="scanner-content">
        {/* Gainers Card */}
        <div className="scanner-card gainers">
          <div className="card-header">
            <span className="card-emoji">🚀</span>
            <span className="card-title">Top Projected Gainers</span>
            {data?.top_bullish && (
              <span className="card-count">{data.top_bullish.length}</span>
            )}
          </div>

          <div className="card-labels">
            <span className="label-rank">#</span>
            <span className="label-stock">Stock</span>
            <span className="label-price">Price</span>
            <span className="label-change">Predicted</span>
            <span className="label-conf">Confidence</span>
          </div>

          <div className="card-list">
            {loading && !data ? (
              <LoadingRows count={5} />
            ) : data?.top_bullish?.length > 0 ? (
              data.top_bullish.map((stock, index) => (
                <StockItem
                  key={stock.symbol}
                  stock={stock}
                  type="gainer"
                  rank={index}
                  onClick={handleStockClick}
                />
              ))
            ) : (
              <div className="card-empty">
                <span>No bullish signals detected</span>
              </div>
            )}
          </div>
        </div>

        {/* Losers Card */}
        <div className="scanner-card losers">
          <div className="card-header">
            <span className="card-emoji">🔻</span>
            <span className="card-title">Top Projected Losers</span>
            {data?.top_bearish && (
              <span className="card-count">{data.top_bearish.length}</span>
            )}
          </div>

          <div className="card-labels">
            <span className="label-rank">#</span>
            <span className="label-stock">Stock</span>
            <span className="label-price">Price</span>
            <span className="label-change">Predicted</span>
            <span className="label-conf">Confidence</span>
          </div>

          <div className="card-list">
            {loading && !data ? (
              <LoadingRows count={5} />
            ) : data?.top_bearish?.length > 0 ? (
              data.top_bearish.map((stock, index) => (
                <StockItem
                  key={stock.symbol}
                  stock={stock}
                  type="loser"
                  rank={index}
                  onClick={handleStockClick}
                />
              ))
            ) : (
              <div className="card-empty">
                <span>No bearish signals detected</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      {data && (
        <div className="scanner-footer">
          {/* TASI Index Info */}
          {marketStatus?.tasi && (
            <>
              <span style={{ fontWeight: 600 }}>
                TASI: {marketStatus.tasi.price?.toLocaleString()}
              </span>
              <span
                style={{
                  color: marketStatus.tasi.daily_change >= 0 ? '#00ffa3' : '#ff4757',
                  fontWeight: 600,
                }}
              >
                {marketStatus.tasi.daily_change >= 0 ? '+' : ''}
                {marketStatus.tasi.daily_change?.toFixed(2)}%
              </span>
              <span className="footer-dot">•</span>
            </>
          )}
          <span>{data.total_scanned} stocks scanned</span>
          <span className="footer-dot">•</span>
          <span>AI-powered predictions</span>
          {data.cache_info?.next_refresh_minutes && (
            <>
              <span className="footer-dot">•</span>
              <span>Auto-refresh in {Math.round(data.cache_info.next_refresh_minutes)}m</span>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default MarketScanner;
