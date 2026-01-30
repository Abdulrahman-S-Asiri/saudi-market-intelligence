// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

/**
 * AIInsightsPanel Component
 *
 * Displays AI-generated analysis summary in natural language
 * with key insights, predictions, and recommendations.
 * Uses BiLSTM + Multi-Head Attention model.
 */

import React from 'react';
import { motion } from 'framer-motion';
import './AIInsightsPanel.css';

const AIInsightsPanel = ({ analysis, mlPrediction, isLoading }) => {
  const generateInsights = () => {
    if (!analysis) return [];

    const insights = [];
    const { signal, indicators, trend, performance, risk } = analysis;

    // Signal insight
    if (signal) {
      const confidence = signal.confidence || 0;
      let sentiment = 'neutral';
      if (signal.action?.toLowerCase().includes('buy')) sentiment = 'bullish';
      if (signal.action?.toLowerCase().includes('sell')) sentiment = 'bearish';

      insights.push({
        type: sentiment,
        icon: sentiment === 'bullish' ? '📈' : sentiment === 'bearish' ? '📉' : '➡️',
        title: `${signal.action} Signal`,
        text: `The AI model suggests a ${signal.action.toLowerCase()} position with ${confidence}% confidence. ${signal.reasons?.[0] || ''}`
      });
    }

    // Trend insight
    if (trend) {
      insights.push({
        type: 'info',
        icon: '🔄',
        title: 'Market Regime',
        text: `Currently in a ${trend.direction || 'neutral'} regime with ${trend.strength || 50}% trend strength.`
      });
    }

    // RSI insight
    if (indicators?.rsi) {
      const rsi = indicators.rsi;
      if (rsi < 30) {
        insights.push({
          type: 'bullish',
          icon: '⚡',
          title: 'Oversold Condition',
          text: `RSI at ${rsi.toFixed(1)} indicates the stock may be oversold. Potential reversal opportunity.`
        });
      } else if (rsi > 70) {
        insights.push({
          type: 'bearish',
          icon: '⚠️',
          title: 'Overbought Condition',
          text: `RSI at ${rsi.toFixed(1)} indicates the stock may be overbought. Consider taking profits.`
        });
      }
    }

    // Performance insight
    if (performance) {
      const sharpe = performance.sharpe_ratio || 0;
      if (sharpe > 1.5) {
        insights.push({
          type: 'bullish',
          icon: '🏆',
          title: 'Strong Risk-Adjusted Returns',
          text: `Sharpe ratio of ${sharpe.toFixed(2)} indicates excellent risk-adjusted performance.`
        });
      } else if (sharpe < 0.5) {
        insights.push({
          type: 'warning',
          icon: '⚠️',
          title: 'Low Risk-Adjusted Returns',
          text: `Sharpe ratio of ${sharpe.toFixed(2)} suggests poor risk-adjusted performance.`
        });
      }
    }

    // ML Prediction insight
    if (mlPrediction) {
      insights.push({
        type: mlPrediction.direction === 'UP' ? 'bullish' : mlPrediction.direction === 'DOWN' ? 'bearish' : 'neutral',
        icon: '🤖',
        title: 'AI Prediction',
        text: `The BiLSTM + Attention model predicts ${mlPrediction.direction} movement with ${mlPrediction.confidence?.toFixed(1)}% confidence.`
      });
    }

    // Risk insight
    if (risk) {
      const var95 = Math.abs(risk.var_95 || 0);
      if (var95 > 5) {
        insights.push({
          type: 'warning',
          icon: '🛡️',
          title: 'Elevated Risk',
          text: `Value at Risk (95%) is ${var95.toFixed(2)}%. Consider position sizing carefully.`
        });
      }
    }

    return insights;
  };

  const insights = generateInsights();

  return (
    <div className="ai-insights-panel">
      <div className="insights-header">
        <div className="insights-title-row">
          <span className="insights-icon">🧠</span>
          <h3 className="insights-title">AI Insights</h3>
        </div>
        <div className="insights-badge">
          <span className="badge-dot" />
          BiLSTM + Multi-Head Attention
        </div>
      </div>

      {isLoading ? (
        <div className="insights-loading">
          <div className="loading-pulse" />
          <span>Analyzing market data...</span>
        </div>
      ) : (
        <div className="insights-list">
          {insights.length === 0 ? (
            <div className="insights-empty">
              <span>No insights available. Run analysis to generate.</span>
            </div>
          ) : (
            insights.map((insight, index) => (
              <motion.div
                key={index}
                className={`insight-card insight-${insight.type}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <span className="insight-icon">{insight.icon}</span>
                <div className="insight-content">
                  <h4 className="insight-title">{insight.title}</h4>
                  <p className="insight-text">{insight.text}</p>
                </div>
              </motion.div>
            ))
          )}
        </div>
      )}

      {/* Model Info */}
      {mlPrediction?.model_info && (
        <div className="model-info">
          <span className="model-info-label">Model:</span>
          <span className="model-info-value">
            {mlPrediction.model_info.architecture || 'BiLSTM+MultiHeadAttention'}
          </span>
          {mlPrediction.model_info.uncertainty && (
            <>
              <span className="model-info-label">Uncertainty:</span>
              <span className="model-info-value">
                ±{(mlPrediction.model_info.uncertainty * 100).toFixed(2)}%
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default AIInsightsPanel;
