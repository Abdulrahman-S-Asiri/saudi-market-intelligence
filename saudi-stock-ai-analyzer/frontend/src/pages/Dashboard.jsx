import React, { useState, useEffect } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import StatCard from '../components/StatCard';
import StockList from '../components/StockList';
import { useStocks, useStockAnalysis, useChartData, useMarketRankings } from '../hooks/useMarketData';
import { createPositionFromSignal } from '../api/client';
import { SparklesIcon, PresentationChartLineIcon, ShieldCheckIcon } from '@heroicons/react/24/solid';

// Chart Component
const TechnicalChart = ({ data }) => {
  const chartContainerRef = React.useRef();
  const chartRef = React.useRef();

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const handleResize = () => {
      chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
    };

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#94a3b8'
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
      },
    });

    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00dc82',
      downColor: '#ff2b2b',
      borderVisible: false,
      wickUpColor: '#00dc82',
      wickDownColor: '#ff2b2b',
    });

    if (data && data.length > 0) {
      const formattedData = data.map(d => ({
        time: d.date,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));
      candleSeries.setData(formattedData);
      chart.timeScale().fitContent();
    }

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data]);

  return <div ref={chartContainerRef} className="w-full h-[400px]" />;
};

// Timeframe mapping for API
const timeframeMap = {
  '1D': '1d', '1W': '5d', '1M': '1mo',
  '3M': '3mo', '6M': '6mo', '1Y': '1y', 'ALL': '2y'
};

const Dashboard = () => {
  const [selectedStock, setSelectedStock] = useState('2222');
  const [selectedTimeframe, setSelectedTimeframe] = useState('6M');
  const [isExecutingTrade, setIsExecutingTrade] = useState(false);
  const [notification, setNotification] = useState(null);

  const apiPeriod = timeframeMap[selectedTimeframe] || '6mo';

  const { data: stocksData } = useStocks();
  const { data: analysisData, isLoading: analysisLoading } = useStockAnalysis(selectedStock, apiPeriod, true);
  const { data: chartDataRes } = useChartData(selectedStock, apiPeriod);
  const { data: rankingsData } = useMarketRankings();

  const stockList = stocksData?.stocks || [];
  const chartData = chartDataRes?.data || [];
  const analysis = analysisData || {};

  const signal = analysis.signal || {};
  const mlPred = analysis.ml_prediction || {};
  const indicators = analysis.indicators || {};
  const risk = analysis.risk || {};

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const handleExecuteTrade = async () => {
    if (!signal.action || signal.action === 'HOLD') return;

    const confirmed = window.confirm(
      `Execute ${signal.action} trade for ${selectedStock} at ${signal.price}?\n\nStop Loss: ${signal.stop_loss}\nTake Profit: ${signal.take_profit}`
    );

    if (!confirmed) return;

    setIsExecutingTrade(true);
    try {
      const position = await createPositionFromSignal(selectedStock);
      setNotification({
        type: 'success',
        message: `Position opened: ${position.position_type} ${position.symbol} @ ${position.entry_price.toFixed(2)}`
      });
    } catch (error) {
      setNotification({
        type: 'error',
        message: error.response?.data?.detail || 'Failed to execute trade'
      });
    } finally {
      setIsExecutingTrade(false);
    }
  };

  const canExecuteTrade = signal.action && signal.action !== 'HOLD' && !isExecutingTrade;

  return (
    <>
      {notification && (
        <div className={`fixed top-20 right-6 z-50 px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 animate-enter ${
          notification.type === 'success' ? 'bg-green-500/90' : 'bg-red-500/90'
        }`}>
          <span className="text-white text-sm font-medium">{notification.message}</span>
          <button onClick={() => setNotification(null)} className="text-white/80 hover:text-white">
            &times;
          </button>
        </div>
      )}

      <main className="px-6 max-w-[1920px] mx-auto grid grid-cols-12 gap-6 animate-enter">
        {/* LEFT SIDEBAR: Stock List */}
        <div className="col-span-12 lg:col-span-3 xl:col-span-2 h-[calc(100vh-100px)] sticky top-24">
          <StockList
            stocks={stockList}
            selectedSymbol={selectedStock}
            onSelect={setSelectedStock}
          />
        </div>

        {/* CENTER CONTENT: Analysis & Chart */}
        <div className="col-span-12 lg:col-span-9 xl:col-span-7 flex flex-col gap-6">
          {/* Header Stats Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard
              title="Current Price"
              value={signal.price || "---"}
              icon={PresentationChartLineIcon}
              isPositive={mlPred.direction === 'UP'}
            />
            <StatCard
              title="AI Confidence"
              value={`${mlPred.confidence || 0}%`}
              change={mlPred.direction}
              isPositive={mlPred.confidence > 70}
              icon={SparklesIcon}
            />
            <StatCard
              title="Risk Level"
              value={risk.value_at_risk_95 ? `${risk.value_at_risk_95}%` : "Low"}
              isPositive={true}
              icon={ShieldCheckIcon}
            />
          </div>

          {/* Main Chart Card */}
          <div className="glass-card p-6 min-h-[500px] flex flex-col">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <span className="text-brand-primary">#{selectedStock}</span> Technical Analysis
                </h2>
                <div className="text-xs text-gray-500 font-mono mt-1">BiLSTM MODEL - REAL-TIME INFERENCE</div>
              </div>
              <div className="flex gap-2">
                {['1D', '1W', '1M', '3M', '6M', '1Y', 'ALL'].map(t => (
                  <button
                    key={t}
                    onClick={() => setSelectedTimeframe(t)}
                    className={`px-3 py-1 text-xs font-bold rounded transition-colors ${
                      selectedTimeframe === t
                        ? 'bg-brand-primary text-white'
                        : 'text-gray-400 hover:text-white hover:bg-white/10'
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex-1 w-full bg-dark-950/30 rounded-lg border border-white/5 overflow-hidden relative">
              {analysisLoading ? (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-8 h-8 border-2 border-brand-primary border-t-transparent rounded-full animate-spin"></div>
                </div>
              ) : (
                <TechnicalChart data={chartData} />
              )}
            </div>
          </div>

          {/* Advanced Metrics Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: 'RSI (14)', value: indicators.rsi || '--' },
              { label: 'MACD', value: indicators.macd || '--' },
              { label: 'Volatility', value: indicators.atr || '--' },
              { label: 'Projected', value: mlPred.prediction ? `${(mlPred.prediction * 100).toFixed(2)}%` : '--' }
            ].map((metric) => (
              <div key={metric.label} className="glass-panel p-4 rounded-xl text-center">
                <div className="text-xs text-gray-500 mb-1">{metric.label}</div>
                <div className="text-lg font-bold font-mono">{metric.value}</div>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT SIDEBAR: AI Insights */}
        <div className="col-span-12 xl:col-span-3 flex flex-col gap-6">
          <div className="glass-card p-6 rounded-2xl">
            <h3 className="text-sm font-bold text-gray-300 uppercase tracking-widest mb-4 flex items-center gap-2">
              <SparklesIcon className="w-4 h-4 text-brand-secondary" />
              AI Recommendation
            </h3>

            <div className="flex flex-col items-center justify-center py-6">
              <div className={`w-32 h-32 rounded-full border-4 flex items-center justify-center mb-4 relative ${signal.action === 'BUY' ? 'border-trade-up text-trade-up shadow-neon-green' :
                  signal.action === 'SELL' ? 'border-trade-down text-trade-down shadow-neon-red' :
                    'border-gray-600 text-gray-400'
                }`}>
                <div className="absolute inset-0 bg-current opacity-10 rounded-full animate-pulse-slow"></div>
                <span className="text-3xl font-bold tracking-tighter">{signal.action || 'WAIT'}</span>
              </div>
              <p className="text-center text-sm text-gray-400 leading-relaxed px-2">
                {signal.reasons && signal.reasons[0] ? signal.reasons[0] : "Analyzing market patterns..."}
              </p>
            </div>

            <div className="space-y-3 mt-4">
              <div className="flex justify-between text-sm py-2 border-b border-white/5">
                <span className="text-gray-500">Entry Target</span>
                <span className="font-mono text-white">{signal.price}</span>
              </div>
              <div className="flex justify-between text-sm py-2 border-b border-white/5">
                <span className="text-gray-500">Stop Loss</span>
                <span className="font-mono text-red-400">{signal.stop_loss || '--'}</span>
              </div>
              <div className="flex justify-between text-sm py-2">
                <span className="text-gray-500">Take Profit</span>
                <span className="font-mono text-green-400">{signal.take_profit || '--'}</span>
              </div>
            </div>

            <button
              onClick={handleExecuteTrade}
              disabled={!canExecuteTrade}
              className={`w-full mt-6 btn-primary flex items-center justify-center gap-2 ${
                !canExecuteTrade ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isExecutingTrade ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Executing...
                </>
              ) : (
                'Execute Trade'
              )}
            </button>
          </div>

          {/* Top Gainers/Losers Mini List */}
          <div className="glass-panel p-4 rounded-2xl flex-1">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">Market Movers</h3>
            <div className="space-y-2">
              {rankingsData?.top_bullish?.slice(0, 5).map(stock => (
                <div key={stock.symbol} className="flex justify-between items-center text-sm p-2 hover:bg-white/5 rounded-lg cursor-pointer" onClick={() => setSelectedStock(stock.symbol)}>
                  <span className="font-bold text-gray-300">{stock.symbol}</span>
                  <span className="font-mono text-trade-up">+{stock.predicted_change}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </>
  );
};

export default Dashboard;
