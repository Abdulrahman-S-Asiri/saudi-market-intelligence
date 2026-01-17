import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';

const PriceChart = ({ data, loading, timeframe, onTimeframeChange }) => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const sma20SeriesRef = useRef(null);
  const sma50SeriesRef = useRef(null);
  const [showVolume, setShowVolume] = useState(true);
  const [showSMA, setShowSMA] = useState(true);

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#888',
        fontFamily: 'Inter, sans-serif',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: 'rgba(0, 212, 255, 0.4)',
          width: 1,
          style: 2,
          labelBackgroundColor: '#00d4ff',
        },
        horzLine: {
          color: 'rgba(0, 212, 255, 0.4)',
          width: 1,
          style: 2,
          labelBackgroundColor: '#00d4ff',
        },
      },
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      handleScroll: { vertTouchDrag: true },
      handleScale: { axisPressedMouseMove: true },
    });

    chartRef.current = chart;

    // Create candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00c853',
      downColor: '#ff1744',
      borderUpColor: '#00c853',
      borderDownColor: '#ff1744',
      wickUpColor: '#00c853',
      wickDownColor: '#ff1744',
    });
    candleSeriesRef.current = candleSeries;

    // Create volume series
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
      scaleMargins: {
        top: 0.85,
        bottom: 0,
      },
    });
    volumeSeriesRef.current = volumeSeries;

    // Create SMA series
    const sma20Series = chart.addLineSeries({
      color: '#ff9f43',
      lineWidth: 1,
      lineStyle: 2,
      title: 'SMA 20',
    });
    sma20SeriesRef.current = sma20Series;

    const sma50Series = chart.addLineSeries({
      color: '#ee5a24',
      lineWidth: 1,
      lineStyle: 2,
      title: 'SMA 50',
    });
    sma50SeriesRef.current = sma50Series;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Update data when it changes
  useEffect(() => {
    if (!candleSeriesRef.current || !data || data.length === 0) return;

    // Format data for candlestick chart
    const candleData = data.map(item => ({
      time: item.date,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    }));

    // Format volume data with colors
    const volumeData = data.map(item => ({
      time: item.date,
      value: item.volume,
      color: item.close >= item.open
        ? 'rgba(0, 200, 83, 0.3)'
        : 'rgba(255, 23, 68, 0.3)',
    }));

    // Format SMA data
    const sma20Data = data
      .filter(item => item.sma_20 != null)
      .map(item => ({
        time: item.date,
        value: item.sma_20,
      }));

    const sma50Data = data
      .filter(item => item.sma_50 != null)
      .map(item => ({
        time: item.date,
        value: item.sma_50,
      }));

    // Set data
    candleSeriesRef.current.setData(candleData);

    if (showVolume && volumeSeriesRef.current) {
      volumeSeriesRef.current.setData(volumeData);
    }

    if (showSMA) {
      if (sma20SeriesRef.current) sma20SeriesRef.current.setData(sma20Data);
      if (sma50SeriesRef.current) sma50SeriesRef.current.setData(sma50Data);
    }

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [data, showVolume, showSMA]);

  // Toggle volume visibility
  useEffect(() => {
    if (volumeSeriesRef.current) {
      volumeSeriesRef.current.applyOptions({
        visible: showVolume,
      });
    }
  }, [showVolume]);

  // Toggle SMA visibility
  useEffect(() => {
    if (sma20SeriesRef.current) {
      sma20SeriesRef.current.applyOptions({ visible: showSMA });
    }
    if (sma50SeriesRef.current) {
      sma50SeriesRef.current.applyOptions({ visible: showSMA });
    }
  }, [showSMA]);

  if (loading || !data || data.length === 0) {
    return (
      <div className="chart-container glass-card">
        <div className="chart-header">
          <h3>Price Chart</h3>
        </div>
        <div className="chart-placeholder">
          {loading ? (
            <div className="chart-loading">
              <div className="loading-spinner"></div>
              <span>Loading chart data...</span>
            </div>
          ) : (
            'No data available'
          )}
        </div>
      </div>
    );
  }

  const latestPrice = data[data.length - 1];
  const prevPrice = data[data.length - 2];
  const priceChange = latestPrice && prevPrice
    ? ((latestPrice.close - prevPrice.close) / prevPrice.close * 100).toFixed(2)
    : 0;
  const isPositive = priceChange >= 0;

  return (
    <div className="chart-container glass-card">
      <div className="chart-header">
        <div className="chart-title-section">
          <h3>Price Chart</h3>
          {latestPrice && (
            <div className="price-info">
              <span className="current-price">{latestPrice.close?.toFixed(2)} SAR</span>
              <span className={`price-change ${isPositive ? 'positive' : 'negative'}`}>
                {isPositive ? '+' : ''}{priceChange}%
              </span>
            </div>
          )}
        </div>
        <div className="chart-controls">
          <button
            className={`chart-toggle ${showVolume ? 'active' : ''}`}
            onClick={() => setShowVolume(!showVolume)}
          >
            Volume
          </button>
          <button
            className={`chart-toggle ${showSMA ? 'active' : ''}`}
            onClick={() => setShowSMA(!showSMA)}
          >
            SMA
          </button>
        </div>
      </div>
      <div
        ref={chartContainerRef}
        className="chart-canvas"
        style={{ height: '400px' }}
      />
      <div className="chart-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ background: '#00c853' }}></span>
          <span>Bullish</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ background: '#ff1744' }}></span>
          <span>Bearish</span>
        </div>
        {showSMA && (
          <>
            <div className="legend-item">
              <span className="legend-line" style={{ background: '#ff9f43' }}></span>
              <span>SMA 20</span>
            </div>
            <div className="legend-item">
              <span className="legend-line" style={{ background: '#ee5a24' }}></span>
              <span>SMA 50</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default PriceChart;
