// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

import React, { useEffect, useRef } from 'react';
import { createChart, ColorType } from 'lightweight-charts';

const MACDChart = ({ data, loading }) => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return;

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
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        visible: false,
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      height: 150,
    });

    chartRef.current = chart;

    // MACD histogram
    const histogramSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
    });

    // MACD line
    const macdSeries = chart.addLineSeries({
      color: '#00d4ff',
      lineWidth: 2,
    });

    // Signal line
    const signalSeries = chart.addLineSeries({
      color: '#ff9f43',
      lineWidth: 2,
    });

    // Format MACD data
    const macdData = data
      .filter(item => item.macd != null)
      .map(item => ({
        time: item.date,
        value: item.macd,
      }));

    const signalData = data
      .filter(item => item.macd_signal != null)
      .map(item => ({
        time: item.date,
        value: item.macd_signal,
      }));

    const histogramData = data
      .filter(item => item.macd != null && item.macd_signal != null)
      .map(item => {
        const histogram = item.macd - item.macd_signal;
        return {
          time: item.date,
          value: histogram,
          color: histogram >= 0
            ? 'rgba(0, 200, 83, 0.6)'
            : 'rgba(255, 23, 68, 0.6)',
        };
      });

    histogramSeries.setData(histogramData);
    macdSeries.setData(macdData);
    signalSeries.setData(signalData);

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data]);

  if (loading || !data || data.length === 0) {
    return (
      <div className="indicator-chart glass-card">
        <div className="indicator-header">
          <h4>MACD (12, 26, 9)</h4>
        </div>
        <div className="indicator-placeholder">
          {loading ? 'Loading...' : 'No data'}
        </div>
      </div>
    );
  }

  const latest = data[data.length - 1];
  const macdValue = latest?.macd;
  const signalValue = latest?.macd_signal;
  const histogram = macdValue && signalValue ? macdValue - signalValue : 0;
  const isBullish = histogram > 0;

  return (
    <div className="indicator-chart glass-card">
      <div className="indicator-header">
        <h4>MACD (12, 26, 9)</h4>
        <div className="indicator-value">
          <span className={isBullish ? 'bullish' : 'bearish'}>
            {histogram?.toFixed(4)}
          </span>
          <span className="indicator-status" style={{ color: isBullish ? '#00c853' : '#ff1744' }}>
            {isBullish ? 'Bullish' : 'Bearish'}
          </span>
        </div>
      </div>
      <div ref={chartContainerRef} className="indicator-canvas" />
      <div className="indicator-legend">
        <span className="legend-macd">MACD</span>
        <span className="legend-signal">Signal</span>
        <span className="legend-histogram">Histogram</span>
      </div>
    </div>
  );
};

export default MACDChart;
