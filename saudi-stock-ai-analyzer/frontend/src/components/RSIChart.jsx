// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

import React, { useEffect, useRef } from 'react';
import { createChart, ColorType } from 'lightweight-charts';

const RSIChart = ({ data, loading }) => {
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

    // RSI line
    const rsiSeries = chart.addLineSeries({
      color: '#00d4ff',
      lineWidth: 2,
    });

    // Overbought line (70)
    const overboughtSeries = chart.addLineSeries({
      color: 'rgba(255, 23, 68, 0.5)',
      lineWidth: 1,
      lineStyle: 2,
    });

    // Oversold line (30)
    const oversoldSeries = chart.addLineSeries({
      color: 'rgba(0, 200, 83, 0.5)',
      lineWidth: 1,
      lineStyle: 2,
    });

    // Neutral line (50)
    const neutralSeries = chart.addLineSeries({
      color: 'rgba(255, 255, 255, 0.2)',
      lineWidth: 1,
      lineStyle: 2,
    });

    // Format RSI data
    const rsiData = data
      .filter(item => item.rsi != null)
      .map(item => ({
        time: item.date,
        value: item.rsi,
      }));

    // Create constant lines
    const timeRange = data.map(item => item.date);
    const overboughtData = timeRange.map(time => ({ time, value: 70 }));
    const oversoldData = timeRange.map(time => ({ time, value: 30 }));
    const neutralData = timeRange.map(time => ({ time, value: 50 }));

    rsiSeries.setData(rsiData);
    overboughtSeries.setData(overboughtData);
    oversoldSeries.setData(oversoldData);
    neutralSeries.setData(neutralData);

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
          <h4>RSI (14)</h4>
        </div>
        <div className="indicator-placeholder">
          {loading ? 'Loading...' : 'No data'}
        </div>
      </div>
    );
  }

  const latestRSI = data[data.length - 1]?.rsi;
  const rsiStatus = latestRSI > 70 ? 'Overbought' : latestRSI < 30 ? 'Oversold' : 'Neutral';
  const rsiColor = latestRSI > 70 ? '#ff1744' : latestRSI < 30 ? '#00c853' : '#00d4ff';

  return (
    <div className="indicator-chart glass-card">
      <div className="indicator-header">
        <h4>RSI (14)</h4>
        <div className="indicator-value">
          <span style={{ color: rsiColor }}>{latestRSI?.toFixed(1)}</span>
          <span className="indicator-status" style={{ color: rsiColor }}>
            {rsiStatus}
          </span>
        </div>
      </div>
      <div ref={chartContainerRef} className="indicator-canvas" />
      <div className="indicator-legend">
        <span className="legend-overbought">70 - Overbought</span>
        <span className="legend-oversold">30 - Oversold</span>
      </div>
    </div>
  );
};

export default RSIChart;
