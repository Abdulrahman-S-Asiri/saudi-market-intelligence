import React, { useState } from 'react';
import { useStocks, useStockAnalysis } from '../hooks/useMarketData';
import { ChartBarIcon, ArrowTrendingUpIcon, ArrowTrendingDownIcon } from '@heroicons/react/24/solid';

const Analysis = () => {
  const [selectedStock, setSelectedStock] = useState('2222');
  const [period, setPeriod] = useState('6mo');

  const { data: stocksData } = useStocks();
  const { data: analysisData, isLoading } = useStockAnalysis(selectedStock, period, true);

  const stockList = stocksData?.stocks || [];
  const analysis = analysisData || {};
  const signal = analysis.signal || {};
  const mlPred = analysis.ml_prediction || {};
  const indicators = analysis.indicators || {};
  const performance = analysis.performance || {};
  const risk = analysis.risk || {};

  return (
    <main className="px-6 max-w-[1920px] mx-auto animate-enter">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-2">Detailed Analysis</h1>
        <p className="text-gray-400">Comprehensive technical and fundamental analysis</p>
      </div>

      {/* Stock Selector */}
      <div className="glass-card p-4 mb-6 flex flex-wrap gap-4 items-center">
        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-400">Stock:</label>
          <select
            value={selectedStock}
            onChange={(e) => setSelectedStock(e.target.value)}
            className="bg-dark-800 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-brand-primary"
          >
            {stockList.map(stock => (
              <option key={stock.symbol} value={stock.symbol}>
                {stock.symbol} - {stock.name}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-400">Period:</label>
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="bg-dark-800 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-brand-primary"
          >
            <option value="1mo">1 Month</option>
            <option value="3mo">3 Months</option>
            <option value="6mo">6 Months</option>
            <option value="1y">1 Year</option>
            <option value="2y">2 Years</option>
          </select>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-8 h-8 border-2 border-brand-primary border-t-transparent rounded-full animate-spin"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Signal Overview */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <ChartBarIcon className="w-5 h-5 text-brand-primary" />
              Signal Overview
            </h3>
            <div className="flex flex-col items-center py-4">
              <div className={`w-24 h-24 rounded-full border-4 flex items-center justify-center mb-4 ${
                signal.action === 'BUY' ? 'border-green-500 text-green-500' :
                signal.action === 'SELL' ? 'border-red-500 text-red-500' :
                'border-gray-500 text-gray-500'
              }`}>
                <span className="text-2xl font-bold">{signal.action || 'HOLD'}</span>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-400 mb-1">Confidence</div>
                <div className="text-xl font-bold text-white">{signal.confidence || 0}%</div>
              </div>
            </div>
            <div className="space-y-2 mt-4">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Price</span>
                <span className="text-white font-mono">{signal.price || '--'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Stop Loss</span>
                <span className="text-red-400 font-mono">{signal.stop_loss || '--'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Take Profit</span>
                <span className="text-green-400 font-mono">{signal.take_profit || '--'}</span>
              </div>
            </div>
          </div>

          {/* ML Prediction */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              {mlPred.direction === 'UP' ? (
                <ArrowTrendingUpIcon className="w-5 h-5 text-green-500" />
              ) : (
                <ArrowTrendingDownIcon className="w-5 h-5 text-red-500" />
              )}
              ML Prediction
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Direction</span>
                <span className={`font-bold ${mlPred.direction === 'UP' ? 'text-green-500' : 'text-red-500'}`}>
                  {mlPred.direction || '--'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Confidence</span>
                <span className="text-white font-mono">{mlPred.confidence || 0}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Prediction</span>
                <span className="text-white font-mono">
                  {mlPred.prediction ? `${(mlPred.prediction * 100).toFixed(2)}%` : '--'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Uncertainty</span>
                <span className="text-white font-mono">
                  {mlPred.uncertainty ? mlPred.uncertainty.toFixed(4) : '--'}
                </span>
              </div>
            </div>
          </div>

          {/* Technical Indicators */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-white mb-4">Technical Indicators</h3>
            <div className="space-y-3">
              {[
                { label: 'RSI (14)', value: indicators.rsi, threshold: 50 },
                { label: 'MACD', value: indicators.macd },
                { label: 'ATR', value: indicators.atr },
                { label: 'SMA 20', value: indicators.sma_20 },
                { label: 'SMA 50', value: indicators.sma_50 },
                { label: 'ADX', value: indicators.adx },
              ].map((ind) => (
                <div key={ind.label} className="flex justify-between items-center text-sm">
                  <span className="text-gray-400">{ind.label}</span>
                  <span className="text-white font-mono">
                    {ind.value !== undefined ? (typeof ind.value === 'number' ? ind.value.toFixed(2) : ind.value) : '--'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-white mb-4">Performance</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Total Return</span>
                <span className={`font-mono ${performance.total_return > 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {performance.total_return ? `${performance.total_return.toFixed(2)}%` : '--'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Volatility</span>
                <span className="text-white font-mono">
                  {performance.volatility ? `${performance.volatility.toFixed(2)}%` : '--'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Sharpe Ratio</span>
                <span className="text-white font-mono">
                  {performance.sharpe_ratio ? performance.sharpe_ratio.toFixed(2) : '--'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Max Drawdown</span>
                <span className="text-red-400 font-mono">
                  {performance.max_drawdown ? `${performance.max_drawdown.toFixed(2)}%` : '--'}
                </span>
              </div>
            </div>
          </div>

          {/* Risk Metrics */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-white mb-4">Risk Metrics</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">VaR (95%)</span>
                <span className="text-white font-mono">
                  {risk.value_at_risk_95 ? `${risk.value_at_risk_95.toFixed(2)}%` : '--'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">VaR (99%)</span>
                <span className="text-white font-mono">
                  {risk.value_at_risk_99 ? `${risk.value_at_risk_99.toFixed(2)}%` : '--'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Beta</span>
                <span className="text-white font-mono">
                  {risk.beta ? risk.beta.toFixed(2) : '--'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">Sortino Ratio</span>
                <span className="text-white font-mono">
                  {risk.sortino_ratio ? risk.sortino_ratio.toFixed(2) : '--'}
                </span>
              </div>
            </div>
          </div>

          {/* Signal Reasons */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-white mb-4">Signal Reasons</h3>
            <div className="space-y-2">
              {signal.reasons && signal.reasons.length > 0 ? (
                signal.reasons.map((reason, idx) => (
                  <div key={idx} className="text-sm text-gray-300 flex items-start gap-2">
                    <span className="text-brand-primary">-</span>
                    <span>{reason}</span>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500">No signal reasons available</p>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
};

export default Analysis;
