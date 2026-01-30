// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

/**
 * Market Signals Component
 * Displays top bullish and bearish stocks based on AI predictions
 */

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './MarketSignals.css';

const API_BASE = 'http://localhost:8000';

// Individual Stock Signal Card
const SignalCard = ({ stock, type, rank, onClick }) => {
  const isBullish = type === 'bullish';

  return (
    <motion.div
      className={`signal-card ${type}`}
      initial={{ opacity: 0, x: isBullish ? -20 : 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: rank * 0.1 }}
      whileHover={{ scale: 1.02, y: -2 }}
      onClick={() => onClick && onClick(stock.symbol)}
    >
      <div className="signal-card-rank">#{rank + 1}</div>
      <div className="signal-card-content">
        <div className="signal-card-header">
          <span className="signal-card-symbol">{stock.symbol}</span>
          <span className={`signal-card-direction ${isBullish ? 'up' : 'down'}`}>
            {isBullish ? '▲' : '▼'} {stock.direction}
          </span>
        </div>
        <div className="signal-card-name">{stock.name}</div>
        <div className="signal-card-details">
          <div className="signal-card-price">
            <span className="label">Price</span>
            <span className="value">{stock.price?.toFixed(2)} SAR</span>
          </div>
          <div className="signal-card-confidence">
            <span className="label">Confidence</span>
            <span className="value">{stock.ml_confidence?.toFixed(1)}%</span>
          </div>
        </div>
        <div className="signal-card-meta">
          <span className="signal-card-sector">{stock.sector}</span>
          <span className={`signal-card-action ${stock.signal_action?.toLowerCase()}`}>
            {stock.signal_action}
          </span>
        </div>
        {/* Confidence bar */}
        <div className="signal-card-bar">
          <motion.div
            className="signal-card-bar-fill"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(stock.ml_confidence || 0, 100)}%` }}
            transition={{ delay: rank * 0.1 + 0.3, duration: 0.5 }}
          />
        </div>
      </div>
    </motion.div>
  );
};

// Loading Skeleton
const SignalSkeleton = ({ count = 5 }) => (
  <div className="signal-skeleton-list">
    {[...Array(count)].map((_, i) => (
      <div key={i} className="signal-skeleton-card">
        <div className="skeleton-rank" />
        <div className="skeleton-content">
          <div className="skeleton-header" />
          <div className="skeleton-name" />
          <div className="skeleton-details" />
        </div>
      </div>
    ))}
  </div>
);

// Main Market Signals Component
const MarketSignals = ({ onStockSelect }) => {
  const [rankings, setRankings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // Fetch market rankings
  const fetchRankings = useCallback(async (forceRefresh = false) => {
    try {
      if (forceRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setError(null);

      const url = `${API_BASE}/api/market-rankings?force_refresh=${forceRefresh}&quick_scan=true`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error('Failed to fetch market rankings');
      }

      const data = await response.json();

      // Handle "updating" status
      if (data.status === 'updating') {
        setError('Rankings are being updated. Please wait...');
        return;
      }

      setRankings(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchRankings();
  }, [fetchRankings]);

  // Format cache age
  const formatCacheAge = (minutes) => {
    if (!minutes) return 'Just now';
    if (minutes < 1) return 'Less than a minute ago';
    if (minutes < 60) return `${Math.round(minutes)} min ago`;
    return `${Math.round(minutes / 60)} hr ago`;
  };

  // Handle stock click
  const handleStockClick = (symbol) => {
    if (onStockSelect) {
      onStockSelect(symbol);
    }
  };

  // Handle refresh
  const handleRefresh = () => {
    fetchRankings(true);
  };

  return (
    <div className="market-signals">
      {/* Header */}
      <div className="market-signals-header">
        <div className="market-signals-title">
          <span className="title-icon">📡</span>
          <h2>Market Signals</h2>
          <span className="title-badge">AI Powered</span>
        </div>
        <div className="market-signals-actions">
          {rankings?.cache_info && (
            <span className="cache-info">
              {rankings.cache_info.from_cache
                ? `Updated ${formatCacheAge(rankings.cache_info.cache_age_minutes)}`
                : 'Fresh data'
              }
            </span>
          )}
          <motion.button
            className="refresh-btn"
            onClick={handleRefresh}
            disabled={refreshing || loading}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className={`refresh-icon ${refreshing ? 'spinning' : ''}`}>↻</span>
            {refreshing ? 'Updating...' : 'Refresh'}
          </motion.button>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="market-signals-error">
          <span>⚠️</span> {error}
        </div>
      )}

      {/* Loading State */}
      {loading && !rankings && (
        <div className="market-signals-loading">
          <div className="signals-grid">
            <div className="signals-column bullish">
              <div className="column-header bullish">
                <span className="column-icon">📈</span>
                <span>Top Bullish</span>
              </div>
              <SignalSkeleton count={5} />
            </div>
            <div className="signals-column bearish">
              <div className="column-header bearish">
                <span className="column-icon">📉</span>
                <span>Top Bearish</span>
              </div>
              <SignalSkeleton count={5} />
            </div>
          </div>
        </div>
      )}

      {/* Rankings Display */}
      {rankings && (
        <div className="signals-grid">
          {/* Bullish Column */}
          <div className="signals-column bullish">
            <div className="column-header bullish">
              <span className="column-icon">📈</span>
              <span>Top Bullish</span>
              <span className="column-count">{rankings.top_bullish?.length || 0}</span>
            </div>
            <AnimatePresence>
              {rankings.top_bullish?.length > 0 ? (
                <div className="signal-list">
                  {rankings.top_bullish.map((stock, index) => (
                    <SignalCard
                      key={stock.symbol}
                      stock={stock}
                      type="bullish"
                      rank={index}
                      onClick={handleStockClick}
                    />
                  ))}
                </div>
              ) : (
                <div className="no-signals">
                  <span>No bullish signals</span>
                </div>
              )}
            </AnimatePresence>
          </div>

          {/* Bearish Column */}
          <div className="signals-column bearish">
            <div className="column-header bearish">
              <span className="column-icon">📉</span>
              <span>Top Bearish</span>
              <span className="column-count">{rankings.top_bearish?.length || 0}</span>
            </div>
            <AnimatePresence>
              {rankings.top_bearish?.length > 0 ? (
                <div className="signal-list">
                  {rankings.top_bearish.map((stock, index) => (
                    <SignalCard
                      key={stock.symbol}
                      stock={stock}
                      type="bearish"
                      rank={index}
                      onClick={handleStockClick}
                    />
                  ))}
                </div>
              ) : (
                <div className="no-signals">
                  <span>No bearish signals</span>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      )}

      {/* Footer Stats */}
      {rankings && (
        <div className="market-signals-footer">
          <span className="footer-stat">
            <strong>{rankings.total_scanned}</strong> stocks analyzed
          </span>
          <span className="footer-divider">•</span>
          <span className="footer-stat">
            Updated: {new Date(rankings.timestamp).toLocaleTimeString()}
          </span>
          {rankings.cache_info?.next_refresh_minutes && (
            <>
              <span className="footer-divider">•</span>
              <span className="footer-stat">
                Next auto-refresh in {Math.round(rankings.cache_info.next_refresh_minutes)} min
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default MarketSignals;
