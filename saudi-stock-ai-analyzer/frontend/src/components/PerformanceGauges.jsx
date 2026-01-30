// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

/**
 * PerformanceGauges Component
 *
 * Animated circular gauges showing key performance metrics
 * like Sharpe Ratio, Win Rate, Max Drawdown, etc.
 */

import React from 'react';
import { motion } from 'framer-motion';
import './PerformanceGauges.css';

const CircularGauge = ({ value, maxValue, label, unit, color, size = 120 }) => {
  const normalizedValue = Math.min(Math.max(value || 0, 0), maxValue);
  const percentage = (normalizedValue / maxValue) * 100;
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const getColorClass = () => {
    if (color) return color;
    if (percentage >= 70) return 'excellent';
    if (percentage >= 50) return 'good';
    if (percentage >= 30) return 'warning';
    return 'poor';
  };

  const colorClass = getColorClass();

  return (
    <div className="gauge-container" style={{ width: size, height: size }}>
      <svg className="gauge-svg" viewBox={`0 0 ${size} ${size}`}>
        {/* Background circle */}
        <circle
          className="gauge-bg"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
        />
        {/* Progress circle */}
        <motion.circle
          className={`gauge-progress gauge-${colorClass}`}
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1, ease: 'easeOut' }}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="gauge-content">
        <motion.span
          className={`gauge-value gauge-value-${colorClass}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          {typeof value === 'number' ? value.toFixed(1) : '—'}
          {unit && <span className="gauge-unit">{unit}</span>}
        </motion.span>
        <span className="gauge-label">{label}</span>
      </div>
    </div>
  );
};

const PerformanceGauges = ({ metrics, backtest }) => {
  const gauges = [
    {
      label: 'Sharpe Ratio',
      value: metrics?.sharpe_ratio || backtest?.sharpe_ratio || 0,
      maxValue: 3,
      unit: '',
      color: (metrics?.sharpe_ratio || 0) > 1.5 ? 'excellent' : (metrics?.sharpe_ratio || 0) > 1 ? 'good' : 'warning'
    },
    {
      label: 'Win Rate',
      value: metrics?.win_rate || backtest?.win_rate || 0,
      maxValue: 100,
      unit: '%',
      color: (metrics?.win_rate || 0) > 70 ? 'excellent' : (metrics?.win_rate || 0) > 55 ? 'good' : 'warning'
    },
    {
      label: 'Total Return',
      value: Math.abs(metrics?.total_return || backtest?.total_return || 0),
      maxValue: 50,
      unit: '%',
      color: (metrics?.total_return || 0) > 0 ? 'excellent' : 'poor'
    },
    {
      label: 'Max Drawdown',
      value: Math.abs(metrics?.max_drawdown || backtest?.max_drawdown || 0),
      maxValue: 30,
      unit: '%',
      color: Math.abs(metrics?.max_drawdown || 0) < 10 ? 'excellent' : Math.abs(metrics?.max_drawdown || 0) < 20 ? 'warning' : 'poor'
    },
    {
      label: 'Sortino Ratio',
      value: metrics?.sortino_ratio || 0,
      maxValue: 4,
      unit: '',
      color: (metrics?.sortino_ratio || 0) > 2 ? 'excellent' : (metrics?.sortino_ratio || 0) > 1 ? 'good' : 'warning'
    },
    {
      label: 'Profit Factor',
      value: backtest?.profit_factor || 0,
      maxValue: 3,
      unit: '',
      color: (backtest?.profit_factor || 0) > 1.5 ? 'excellent' : (backtest?.profit_factor || 0) > 1 ? 'good' : 'poor'
    }
  ];

  return (
    <div className="performance-gauges">
      <div className="gauges-header">
        <h3 className="gauges-title">Performance Metrics</h3>
        <span className="gauges-subtitle">Real-time analysis</span>
      </div>

      <div className="gauges-grid">
        {gauges.map((gauge, index) => (
          <motion.div
            key={gauge.label}
            className="gauge-wrapper"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <CircularGauge
              value={gauge.value}
              maxValue={gauge.maxValue}
              label={gauge.label}
              unit={gauge.unit}
              color={gauge.color}
            />
          </motion.div>
        ))}
      </div>

      {/* Summary Stats */}
      <div className="gauges-summary">
        <div className="summary-item">
          <span className="summary-label">Total Trades</span>
          <span className="summary-value">{backtest?.total_trades || '—'}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Volatility</span>
          <span className="summary-value">{metrics?.volatility?.toFixed(1) || '—'}%</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">VaR (95%)</span>
          <span className="summary-value">{metrics?.var_95?.toFixed(2) || '—'}%</span>
        </div>
      </div>
    </div>
  );
};

export default PerformanceGauges;
