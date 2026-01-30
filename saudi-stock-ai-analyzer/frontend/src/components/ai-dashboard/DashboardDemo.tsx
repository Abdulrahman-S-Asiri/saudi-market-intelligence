/**
 * AI Dashboard Demo
 * =================
 * Complete demonstration of all AI-native components working together.
 *
 * This page showcases the TFT model visualization capabilities
 * with sample data matching the backend output format.
 *
 * @author Claude AI / Abdulrahman Asiri
 * @version 1.0.0
 */

import React from 'react';
import { motion } from 'framer-motion';
import { UncertaintyChart } from './UncertaintyChart';
import { AttentionTimeline } from './AttentionTimeline';
import { RiskAdjustedBadge, RiskAdjustedCard } from './RiskAdjustedBadge';

// ============================================================================
// SAMPLE DATA (Matching TFT Backend Output)
// ============================================================================

// Historical price data (60 days lookback)
const generateHistoricalData = () => {
  const data = [];
  let price = 25.50; // Aramco-like starting price

  for (let i = 60; i >= 1; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);

    // Random walk with slight upward bias
    price = price + (Math.random() - 0.48) * 0.3;
    price = Math.max(23, Math.min(28, price)); // Keep in realistic range

    data.push({
      date: date.toISOString().split('T')[0],
      price: parseFloat(price.toFixed(2)),
      open: parseFloat((price - Math.random() * 0.2).toFixed(2)),
      high: parseFloat((price + Math.random() * 0.3).toFixed(2)),
      low: parseFloat((price - Math.random() * 0.3).toFixed(2)),
      volume: Math.floor(50000000 + Math.random() * 20000000),
    });
  }
  return data;
};

// Forecast data (5 days ahead with uncertainty)
const generateForecastData = (lastPrice: number) => {
  const data = [];
  let median = lastPrice;

  for (let i = 1; i <= 5; i++) {
    const date = new Date();
    date.setDate(date.getDate() + i);

    // Model predicts slight upward movement with increasing uncertainty
    median = median + (Math.random() - 0.45) * 0.2;
    const uncertainty = 0.02 + i * 0.008; // Uncertainty grows with horizon

    data.push({
      date: date.toISOString().split('T')[0],
      median: parseFloat(median.toFixed(2)),
      lower: parseFloat((median * (1 - uncertainty)).toFixed(2)),
      upper: parseFloat((median * (1 + uncertainty)).toFixed(2)),
      confidence: parseFloat((0.85 - i * 0.08).toFixed(2)), // Confidence decreases
    });
  }
  return data;
};

// Attention weights (60 days)
const generateAttentionData = () => {
  const data = [];

  for (let i = 60; i >= 1; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);

    // Create realistic attention pattern:
    // - Recent days have higher attention
    // - Some spikes for "important" days
    let weight = 0.1 + (60 - i) / 100; // Base: recent = higher

    // Add random spikes
    if (Math.random() > 0.85) {
      weight = Math.min(1, weight + 0.3 + Math.random() * 0.3);
    }

    // Make last 5 days more important
    if (i <= 5) {
      weight = Math.min(1, weight + 0.2);
    }

    data.push({
      date: date.toISOString().split('T')[0],
      weight: parseFloat(weight.toFixed(3)),
      dayIndex: i,
      event: weight > 0.7 ? 'High volatility detected' : undefined,
    });
  }
  return data;
};

// Sample stock data
const sampleStocks = [
  {
    stockCode: '2222',
    stockName: 'Saudi Aramco',
    currentPrice: 25.80,
    predictedReturn: 0.018, // 1.8%
    confidenceScore: 0.78,
  },
  {
    stockCode: '1120',
    stockName: 'Al Rajhi Bank',
    currentPrice: 107.20,
    predictedReturn: 0.022, // 2.2%
    confidenceScore: 0.45,
  },
  {
    stockCode: '2010',
    stockName: 'SABIC',
    currentPrice: 56.85,
    predictedReturn: -0.012, // -1.2%
    confidenceScore: 0.72,
  },
  {
    stockCode: '7010',
    stockName: 'STC',
    currentPrice: 44.40,
    predictedReturn: 0.008, // 0.8%
    confidenceScore: 0.62,
  },
];

// ============================================================================
// DEMO PAGE COMPONENT
// ============================================================================

export const DashboardDemo: React.FC = () => {
  // Generate sample data
  const historicalData = generateHistoricalData();
  const lastPrice = historicalData[historicalData.length - 1].price;
  const forecastData = generateForecastData(lastPrice);
  const attentionData = generateAttentionData();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-1/4 w-1/2 h-1/2 bg-purple-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 -right-1/4 w-1/2 h-1/2 bg-cyan-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 container mx-auto px-6 py-8">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600">
              <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">
                Saudi Market AI Dashboard
              </h1>
              <p className="text-slate-400 mt-1">
                TFT-Powered Stock Predictions with Uncertainty Quantification
              </p>
            </div>
          </div>

          {/* Quick Stats Bar */}
          <div className="flex items-center gap-6 p-4 backdrop-blur-xl bg-slate-800/30 rounded-xl border border-slate-700/30">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-sm text-slate-400">Model Status:</span>
              <span className="text-sm font-medium text-green-400">Active</span>
            </div>
            <div className="h-4 w-px bg-slate-700" />
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Last Updated:</span>
              <span className="text-sm font-medium text-white">
                {new Date().toLocaleTimeString()}
              </span>
            </div>
            <div className="h-4 w-px bg-slate-700" />
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Stocks Analyzed:</span>
              <span className="text-sm font-medium text-cyan-400">10</span>
            </div>
          </div>
        </motion.header>

        {/* Main Chart Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <UncertaintyChart
            historicalData={historicalData}
            forecastData={forecastData}
            stockCode="2222"
            stockName="Saudi Aramco"
            currency="SAR"
            height={400}
          />
        </motion.section>

        {/* Attention Timeline */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-8"
        >
          <AttentionTimeline
            data={attentionData}
            height={56}
            colorScheme="purple"
            title="AI Temporal Attention - What the model focused on"
          />
        </motion.section>

        {/* Signal Badges Demo */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mb-8"
        >
          <div className="backdrop-blur-xl bg-slate-900/60 border border-slate-700/50 rounded-2xl p-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
              Risk-Adjusted Signal Badges
            </h2>

            <div className="flex flex-wrap gap-4">
              {/* All signal types */}
              <RiskAdjustedBadge predictedReturn={0.025} confidenceScore={0.85} size="lg" showDetails />
              <RiskAdjustedBadge predictedReturn={0.015} confidenceScore={0.55} size="lg" showDetails />
              <RiskAdjustedBadge predictedReturn={0.020} confidenceScore={0.30} size="lg" showDetails />
              <RiskAdjustedBadge predictedReturn={0.005} confidenceScore={0.50} size="lg" showDetails />
              <RiskAdjustedBadge predictedReturn={-0.008} confidenceScore={0.45} size="lg" showDetails />
              <RiskAdjustedBadge predictedReturn={-0.018} confidenceScore={0.75} size="lg" showDetails />
            </div>
          </div>
        </motion.section>

        {/* Stock Cards Grid */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            Top Stock Signals
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {sampleStocks.map((stock, index) => (
              <motion.div
                key={stock.stockCode}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 + index * 0.1 }}
              >
                <RiskAdjustedCard
                  stockCode={stock.stockCode}
                  stockName={stock.stockName}
                  currentPrice={stock.currentPrice}
                  predictedReturn={stock.predictedReturn}
                  confidenceScore={stock.confidenceScore}
                  currency="SAR"
                />
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="mt-12 pt-8 border-t border-slate-800"
        >
          <div className="flex items-center justify-between text-sm text-slate-500">
            <div className="flex items-center gap-2">
              <span>Powered by</span>
              <span className="font-semibold text-purple-400">Temporal Fusion Transformer</span>
            </div>
            <div className="flex items-center gap-4">
              <span>Quantile Regression: 10%, 50%, 90%</span>
              <span className="text-slate-600">|</span>
              <span>Sharpe-Optimized Training</span>
            </div>
          </div>
        </motion.footer>
      </div>
    </div>
  );
};

export default DashboardDemo;
