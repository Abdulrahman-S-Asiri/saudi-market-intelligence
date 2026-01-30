// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const SignalPanel = ({ signal, lstmPrediction, loading, onCreatePosition, symbol }) => {
  const [animateSignal, setAnimateSignal] = useState(false);

  useEffect(() => {
    if (signal) {
      setAnimateSignal(true);
      const timer = setTimeout(() => setAnimateSignal(false), 1000);
      return () => clearTimeout(timer);
    }
  }, [signal?.action]);

  if (!signal && !loading) return null;

  const getSignalColor = (action) => {
    switch (action) {
      case 'BUY': return '#00c853';
      case 'SELL': return '#ff1744';
      default: return '#ffc107';
    }
  };

  const getSignalGradient = (action) => {
    switch (action) {
      case 'BUY': return 'linear-gradient(135deg, #00c853 0%, #00a844 100%)';
      case 'SELL': return 'linear-gradient(135deg, #ff1744 0%, #d50000 100%)';
      default: return 'linear-gradient(135deg, #ffc107 0%, #ff9800 100%)';
    }
  };

  const getSignalIcon = (action) => {
    switch (action) {
      case 'BUY': return '/\\';
      case 'SELL': return '\\/';
      default: return '--';
    }
  };

  // Circular gauge component
  const ConfidenceGauge = ({ value, color }) => {
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const progress = (value / 100) * circumference;

    return (
      <div className="confidence-gauge">
        <svg width="120" height="120" viewBox="0 0 120 120">
          {/* Background circle */}
          <circle
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth="8"
          />
          {/* Progress circle */}
          <motion.circle
            cx="60"
            cy="60"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference - progress }}
            transition={{ duration: 1, ease: 'easeOut' }}
            transform="rotate(-90 60 60)"
            style={{ filter: `drop-shadow(0 0 10px ${color})` }}
          />
        </svg>
        <div className="gauge-value">
          <motion.span
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
          >
            {value}%
          </motion.span>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="signal-panel glass-card">
        <h3>Trading Signal</h3>
        <div className="signal-loading">
          <div className="loading-spinner"></div>
          <span>Analyzing market data...</span>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      className="signal-panel glass-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="signal-panel-header">
        <h3>Trading Signal</h3>
        <span className="signal-time">
          {new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>

      {/* Main Signal Badge */}
      <motion.div
        className={`signal-badge-large ${animateSignal ? 'pulse' : ''}`}
        style={{ background: getSignalGradient(signal.action) }}
        animate={animateSignal ? { scale: [1, 1.05, 1] } : {}}
        transition={{ duration: 0.5 }}
      >
        <motion.div
          className="signal-icon-large"
          animate={animateSignal ? { rotate: [0, 10, -10, 0] } : {}}
        >
          {getSignalIcon(signal.action)}
        </motion.div>
        <span className="signal-action-large">{signal.action}</span>
      </motion.div>

      {/* Confidence Gauge */}
      <div className="confidence-section">
        <h4>Overall Confidence</h4>
        <ConfidenceGauge
          value={signal.confidence}
          color={getSignalColor(signal.action)}
        />
      </div>

      {/* LSTM vs Technical Breakdown */}
      <div className="signal-breakdown">
        <h4>Signal Breakdown</h4>

        {/* LSTM Prediction */}
        {lstmPrediction && (
          <motion.div
            className="breakdown-item"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="breakdown-header">
              <span className="breakdown-label">AI Prediction (LSTM)</span>
              <span className="breakdown-weight">40%</span>
            </div>
            <div className="breakdown-content">
              <span
                className="prediction-badge"
                style={{
                  backgroundColor: `${getSignalColor(
                    lstmPrediction.direction === 'UP' ? 'BUY' : lstmPrediction.direction === 'DOWN' ? 'SELL' : 'HOLD'
                  )}20`,
                  color: getSignalColor(
                    lstmPrediction.direction === 'UP' ? 'BUY' : lstmPrediction.direction === 'DOWN' ? 'SELL' : 'HOLD'
                  ),
                }}
              >
                {lstmPrediction.direction}
              </span>
              <div className="mini-bar">
                <div
                  className="mini-bar-fill"
                  style={{
                    width: `${lstmPrediction.confidence}%`,
                    backgroundColor: getSignalColor(
                      lstmPrediction.direction === 'UP' ? 'BUY' : lstmPrediction.direction === 'DOWN' ? 'SELL' : 'HOLD'
                    ),
                  }}
                />
              </div>
              <span className="breakdown-value">{lstmPrediction.confidence}%</span>
            </div>
          </motion.div>
        )}

        {/* Technical Indicators */}
        <motion.div
          className="breakdown-item"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="breakdown-header">
            <span className="breakdown-label">Technical Analysis</span>
            <span className="breakdown-weight">60%</span>
          </div>
          <div className="breakdown-content">
            <span className="breakdown-indicators">RSI, MACD, SMA</span>
            <div className="mini-bar">
              <div
                className="mini-bar-fill"
                style={{
                  width: `${Math.min(signal.confidence + 10, 100)}%`,
                  backgroundColor: getSignalColor(signal.action),
                }}
              />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Analysis Reasons */}
      <div className="signal-reasons">
        <h4>Analysis Details</h4>
        <AnimatePresence>
          <ul>
            {signal.reasons?.map((reason, index) => (
              <motion.li
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <span className="reason-bullet">+</span>
                {reason}
              </motion.li>
            ))}
          </ul>
        </AnimatePresence>
      </div>

      {/* Price Target */}
      <div className="price-target">
        <div className="target-item">
          <span className="target-label">Entry Price</span>
          <span className="target-value">{signal.price?.toFixed(2)} SAR</span>
        </div>
      </div>

      {/* Open Position Button - Only show for BUY signals */}
      {signal.action === 'BUY' && onCreatePosition && (
        <motion.div
          className="signal-action"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <button
            className="btn-open-position"
            onClick={() => onCreatePosition(symbol, signal.price)}
            style={{
              width: '100%',
              padding: '14px 20px',
              marginTop: '16px',
              background: 'linear-gradient(135deg, #00c853 0%, #009624 100%)',
              border: 'none',
              borderRadius: '10px',
              color: '#fff',
              fontSize: '14px',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = '0 4px 20px rgba(0, 200, 83, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = 'none';
            }}
          >
            <span>+</span>
            Open Position from Signal
          </button>
        </motion.div>
      )}
    </motion.div>
  );
};

export default SignalPanel;
