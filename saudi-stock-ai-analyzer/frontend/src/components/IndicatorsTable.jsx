// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

import React from 'react';

const IndicatorsTable = ({ indicators, supportResistance }) => {
  if (!indicators) return null;

  const getRsiColor = (value) => {
    if (value >= 70) return '#ff1744';
    if (value <= 30) return '#00c853';
    return '#ffc107';
  };

  const getRsiStatus = (value) => {
    if (value >= 70) return 'Overbought';
    if (value <= 30) return 'Oversold';
    return 'Neutral';
  };

  return (
    <div className="indicators-table">
      <h3>Technical Indicators</h3>

      <table>
        <thead>
          <tr>
            <th>Indicator</th>
            <th>Value</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>RSI (14)</td>
            <td style={{ color: getRsiColor(indicators.rsi) }}>{indicators.rsi}</td>
            <td>{getRsiStatus(indicators.rsi)}</td>
          </tr>
          <tr>
            <td>MACD</td>
            <td style={{ color: indicators.macd > 0 ? '#00c853' : '#ff1744' }}>
              {indicators.macd?.toFixed(4)}
            </td>
            <td>{indicators.macd > indicators.macd_signal ? 'Bullish' : 'Bearish'}</td>
          </tr>
          <tr>
            <td>SMA 20</td>
            <td>{indicators.sma_20}</td>
            <td>{indicators.sma_20 > indicators.sma_50 ? 'Above SMA50' : 'Below SMA50'}</td>
          </tr>
          <tr>
            <td>SMA 50</td>
            <td>{indicators.sma_50}</td>
            <td>-</td>
          </tr>
        </tbody>
      </table>

      {supportResistance && (
        <div className="support-resistance">
          <h4>Support & Resistance</h4>
          <div className="sr-levels">
            <div className="sr-item resistance">
              <span className="sr-label">R2</span>
              <span className="sr-value">{supportResistance.resistance_2} SAR</span>
            </div>
            <div className="sr-item resistance">
              <span className="sr-label">R1</span>
              <span className="sr-value">{supportResistance.resistance_1} SAR</span>
            </div>
            <div className="sr-item pivot">
              <span className="sr-label">Pivot</span>
              <span className="sr-value">{supportResistance.pivot} SAR</span>
            </div>
            <div className="sr-item support">
              <span className="sr-label">S1</span>
              <span className="sr-value">{supportResistance.support_1} SAR</span>
            </div>
            <div className="sr-item support">
              <span className="sr-label">S2</span>
              <span className="sr-value">{supportResistance.support_2} SAR</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IndicatorsTable;
