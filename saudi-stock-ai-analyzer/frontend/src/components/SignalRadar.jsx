// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

/**
 * SignalRadar Component
 *
 * Circular radar visualization showing indicator values
 * with animated sweep and signal strength indication.
 */

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import './SignalRadar.css';

const SignalRadar = ({ indicators, signal, confidence }) => {
  const radarData = useMemo(() => {
    if (!indicators) return [];

    return [
      { label: 'RSI', value: indicators.rsi || 50, max: 100, color: getIndicatorColor(indicators.rsi, 30, 70) },
      { label: 'MACD', value: normalizeMACD(indicators.macd), max: 100, color: indicators.macd > 0 ? 'buy' : 'sell' },
      { label: 'ADX', value: indicators.adx || 25, max: 100, color: indicators.adx > 25 ? 'buy' : 'neutral' },
      { label: 'Stoch', value: indicators.stochastic_k || 50, max: 100, color: getIndicatorColor(indicators.stochastic_k, 20, 80) },
      { label: 'Vol', value: normalizeVolume(indicators.volume_ratio), max: 100, color: 'neutral' },
      { label: 'ATR', value: normalizeATR(indicators.atr), max: 100, color: 'neutral' },
    ];
  }, [indicators]);

  function getIndicatorColor(value, low, high) {
    if (value <= low) return 'buy';
    if (value >= high) return 'sell';
    return 'neutral';
  }

  function normalizeMACD(macd) {
    if (!macd) return 50;
    return Math.min(100, Math.max(0, 50 + macd * 500));
  }

  function normalizeVolume(ratio) {
    if (!ratio) return 50;
    return Math.min(100, ratio * 50);
  }

  function normalizeATR(atr) {
    if (!atr) return 50;
    return Math.min(100, atr * 20);
  }

  const getSignalClass = () => {
    if (!signal) return 'neutral';
    const s = signal.toLowerCase();
    if (s.includes('buy')) return 'buy';
    if (s.includes('sell')) return 'sell';
    return 'neutral';
  };

  const signalClass = getSignalClass();
  const numPoints = radarData.length;
  const angleStep = (2 * Math.PI) / numPoints;
  const centerX = 150;
  const centerY = 150;
  const maxRadius = 100;

  // Generate polygon points
  const polygonPoints = radarData.map((item, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const radius = (item.value / item.max) * maxRadius;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className={`signal-radar signal-radar-${signalClass}`}>
      <div className="radar-header">
        <h3 className="radar-title">Signal Radar</h3>
        <div className={`radar-signal-badge signal-${signalClass}`}>
          {signal || 'ANALYZING'}
        </div>
      </div>

      <div className="radar-container">
        <svg viewBox="0 0 300 300" className="radar-svg">
          {/* Background circles */}
          {[0.25, 0.5, 0.75, 1].map((scale, i) => (
            <circle
              key={i}
              cx={centerX}
              cy={centerY}
              r={maxRadius * scale}
              className="radar-circle"
            />
          ))}

          {/* Axis lines */}
          {radarData.map((_, i) => {
            const angle = i * angleStep - Math.PI / 2;
            const x = centerX + maxRadius * Math.cos(angle);
            const y = centerY + maxRadius * Math.sin(angle);
            return (
              <line
                key={i}
                x1={centerX}
                y1={centerY}
                x2={x}
                y2={y}
                className="radar-axis"
              />
            );
          })}

          {/* Data polygon */}
          <motion.polygon
            points={polygonPoints}
            className={`radar-polygon radar-polygon-${signalClass}`}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          />

          {/* Data points */}
          {radarData.map((item, i) => {
            const angle = i * angleStep - Math.PI / 2;
            const radius = (item.value / item.max) * maxRadius;
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);
            return (
              <motion.circle
                key={i}
                cx={x}
                cy={y}
                r={6}
                className={`radar-point radar-point-${item.color}`}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: i * 0.1 }}
              />
            );
          })}

          {/* Labels */}
          {radarData.map((item, i) => {
            const angle = i * angleStep - Math.PI / 2;
            const labelRadius = maxRadius + 25;
            const x = centerX + labelRadius * Math.cos(angle);
            const y = centerY + labelRadius * Math.sin(angle);
            return (
              <text
                key={i}
                x={x}
                y={y}
                className="radar-label"
                textAnchor="middle"
                dominantBaseline="middle"
              >
                {item.label}
              </text>
            );
          })}

          {/* Center sweep animation */}
          <motion.line
            x1={centerX}
            y1={centerY}
            x2={centerX}
            y2={centerY - maxRadius}
            className="radar-sweep"
            animate={{ rotate: 360 }}
            transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
            style={{ transformOrigin: `${centerX}px ${centerY}px` }}
          />
        </svg>

        {/* Center display */}
        <div className="radar-center">
          <div className="radar-confidence">{confidence || 0}%</div>
          <div className="radar-confidence-label">Confidence</div>
        </div>
      </div>

      {/* Indicator values */}
      <div className="radar-indicators">
        {radarData.map((item, i) => (
          <div key={i} className="radar-indicator-item">
            <span className="indicator-label">{item.label}</span>
            <span className={`indicator-value indicator-${item.color}`}>
              {item.value.toFixed(1)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SignalRadar;
