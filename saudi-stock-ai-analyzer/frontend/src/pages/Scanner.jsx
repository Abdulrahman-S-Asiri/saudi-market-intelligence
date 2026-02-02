import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useMarketRankings } from '../hooks/useMarketData';
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon, ArrowPathIcon } from '@heroicons/react/24/solid';

const Scanner = () => {
  const navigate = useNavigate();
  const { data: rankingsData, isLoading, refetch, isFetching } = useMarketRankings();

  const topBullish = rankingsData?.top_bullish || [];
  const topBearish = rankingsData?.top_bearish || [];
  const totalScanned = rankingsData?.total_scanned || 0;
  const timestamp = rankingsData?.timestamp;

  const handleStockClick = (symbol) => {
    navigate(`/?stock=${symbol}`);
  };

  return (
    <main className="px-6 max-w-[1920px] mx-auto animate-enter">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">Market Scanner</h1>
          <p className="text-gray-400">AI-powered market analysis - Top movers and signals</p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="btn-primary flex items-center gap-2"
        >
          <ArrowPathIcon className={`w-4 h-4 ${isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Stats Bar */}
      <div className="glass-card p-4 mb-6 flex flex-wrap gap-6 items-center">
        <div className="text-sm">
          <span className="text-gray-400">Stocks Scanned:</span>
          <span className="text-white font-mono ml-2">{totalScanned}</span>
        </div>
        <div className="text-sm">
          <span className="text-gray-400">Last Updated:</span>
          <span className="text-white font-mono ml-2">
            {timestamp ? new Date(timestamp).toLocaleTimeString() : '--'}
          </span>
        </div>
        <div className="text-sm">
          <span className="text-gray-400">Method:</span>
          <span className="text-brand-primary font-mono ml-2">{rankingsData?.confidence_method || 'mc_dropout'}</span>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-8 h-8 border-2 border-brand-primary border-t-transparent rounded-full animate-spin"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Bullish */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <ArrowTrendingUpIcon className="w-5 h-5 text-green-500" />
              Top Bullish Signals
            </h3>
            <div className="space-y-3">
              {topBullish.length > 0 ? (
                topBullish.map((stock, idx) => (
                  <div
                    key={stock.symbol}
                    onClick={() => handleStockClick(stock.symbol)}
                    className="flex items-center justify-between p-4 bg-white/5 rounded-xl hover:bg-white/10 cursor-pointer transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center text-green-500 font-bold text-sm">
                        #{idx + 1}
                      </div>
                      <div>
                        <div className="font-bold text-white">{stock.symbol}</div>
                        <div className="text-xs text-gray-400">{stock.name}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-green-500 font-mono font-bold">
                        +{stock.predicted_change?.toFixed(2)}%
                      </div>
                      <div className="text-xs text-gray-400">
                        {stock.ml_confidence?.toFixed(0)}% conf
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 text-center py-4">No bullish signals found</p>
              )}
            </div>
          </div>

          {/* Top Bearish */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <ArrowTrendingDownIcon className="w-5 h-5 text-red-500" />
              Top Bearish Signals
            </h3>
            <div className="space-y-3">
              {topBearish.length > 0 ? (
                topBearish.map((stock, idx) => (
                  <div
                    key={stock.symbol}
                    onClick={() => handleStockClick(stock.symbol)}
                    className="flex items-center justify-between p-4 bg-white/5 rounded-xl hover:bg-white/10 cursor-pointer transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center text-red-500 font-bold text-sm">
                        #{idx + 1}
                      </div>
                      <div>
                        <div className="font-bold text-white">{stock.symbol}</div>
                        <div className="text-xs text-gray-400">{stock.name}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-red-500 font-mono font-bold">
                        {stock.predicted_change?.toFixed(2)}%
                      </div>
                      <div className="text-xs text-gray-400">
                        {stock.ml_confidence?.toFixed(0)}% conf
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 text-center py-4">No bearish signals found</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* All Scanned Stocks Table */}
      {rankingsData?.stocks_analyzed && (
        <div className="glass-card p-6 mt-6">
          <h3 className="text-lg font-bold text-white mb-4">Scanned Stocks</h3>
          <div className="flex flex-wrap gap-2">
            {rankingsData.stocks_analyzed.map(symbol => (
              <button
                key={symbol}
                onClick={() => handleStockClick(symbol)}
                className="px-3 py-1 text-xs bg-white/5 hover:bg-white/10 rounded-lg text-gray-300 hover:text-white transition-colors"
              >
                {symbol}
              </button>
            ))}
          </div>
        </div>
      )}
    </main>
  );
};

export default Scanner;
