/**
 * UncertaintyChart Component
 * ==========================
 * Visualizes probabilistic forecasts with uncertainty bands.
 *
 * Features:
 * - Historical data as line chart
 * - Forecast area showing 10th-90th percentile range
 * - Median (50th) as dashed neon line
 * - Smooth fade-in animation with Framer Motion
 *
 * @author Claude AI / Abdulrahman Asiri
 * @version 1.0.0
 */

import React, { useMemo } from 'react';
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { motion } from 'framer-motion';

// ============================================================================
// TYPES
// ============================================================================

interface HistoricalDataPoint {
  date: string;
  price: number;
  open?: number;
  high?: number;
  low?: number;
  volume?: number;
}

interface ForecastDataPoint {
  date: string;
  median: number;      // 50th percentile
  lower: number;       // 10th percentile
  upper: number;       // 90th percentile
  confidence?: number;
}

interface UncertaintyChartProps {
  historicalData: HistoricalDataPoint[];
  forecastData: ForecastDataPoint[];
  stockCode: string;
  stockName?: string;
  currency?: string;
  showGrid?: boolean;
  height?: number;
  animationDuration?: number;
}

// ============================================================================
// CUSTOM TOOLTIP
// ============================================================================

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
  currency: string;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({
  active,
  payload,
  label,
  currency,
}) => {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0]?.payload;
  const isForecast = data?.median !== undefined;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="backdrop-blur-xl bg-slate-900/80 border border-slate-700/50
                 rounded-xl p-4 shadow-2xl shadow-purple-500/10"
    >
      <p className="text-xs text-slate-400 mb-2">{label}</p>

      {isForecast ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-sm text-slate-300">Median:</span>
            <span className="text-sm font-bold text-cyan-400">
              {currency} {data.median?.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-purple-500/50" />
            <span className="text-xs text-slate-400">Range:</span>
            <span className="text-xs text-purple-300">
              {currency} {data.lower?.toFixed(2)} - {data.upper?.toFixed(2)}
            </span>
          </div>
          {data.confidence !== undefined && (
            <div className="mt-2 pt-2 border-t border-slate-700/50">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Confidence</span>
                <span className={`text-xs font-medium ${
                  data.confidence > 0.7 ? 'text-green-400' :
                  data.confidence > 0.4 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {(data.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-400" />
          <span className="text-sm text-slate-300">Price:</span>
          <span className="text-sm font-bold text-blue-400">
            {currency} {data.price?.toFixed(2)}
          </span>
        </div>
      )}
    </motion.div>
  );
};

// ============================================================================
// GRADIENT DEFINITIONS
// ============================================================================

const GradientDefs: React.FC = () => (
  <defs>
    {/* Forecast uncertainty gradient */}
    <linearGradient id="uncertaintyGradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stopColor="#a855f7" stopOpacity={0.6} />
      <stop offset="50%" stopColor="#8b5cf6" stopOpacity={0.3} />
      <stop offset="100%" stopColor="#6366f1" stopOpacity={0.1} />
    </linearGradient>

    {/* Historical line gradient */}
    <linearGradient id="historicalGradient" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.8} />
      <stop offset="100%" stopColor="#06b6d4" stopOpacity={1} />
    </linearGradient>

    {/* Neon glow filter */}
    <filter id="neonGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3" result="coloredBlur" />
      <feMerge>
        <feMergeNode in="coloredBlur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>

    {/* Forecast area glow */}
    <filter id="areaGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4" result="blur" />
      <feFlood floodColor="#a855f7" floodOpacity="0.3" />
      <feComposite in2="blur" operator="in" />
      <feMerge>
        <feMergeNode />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>
);

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const UncertaintyChart: React.FC<UncertaintyChartProps> = ({
  historicalData,
  forecastData,
  stockCode,
  stockName = 'Stock',
  currency = 'SAR',
  showGrid = true,
  height = 400,
  animationDuration = 1.5,
}) => {
  // Combine historical and forecast data for the chart
  const chartData = useMemo(() => {
    const historical = historicalData.map((d) => ({
      date: d.date,
      price: d.price,
      isHistorical: true,
    }));

    const forecast = forecastData.map((d) => ({
      date: d.date,
      median: d.median,
      lower: d.lower,
      upper: d.upper,
      range: [d.lower, d.upper],
      confidence: d.confidence,
      isForecast: true,
    }));

    // Add transition point (last historical = first forecast base)
    if (historical.length > 0 && forecast.length > 0) {
      const lastPrice = historical[historical.length - 1].price;
      forecast[0] = {
        ...forecast[0],
        transitionPrice: lastPrice,
      };
    }

    return [...historical, ...forecast];
  }, [historicalData, forecastData]);

  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    const allValues = [
      ...historicalData.map((d) => d.price),
      ...forecastData.map((d) => d.lower),
      ...forecastData.map((d) => d.upper),
    ];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const padding = (max - min) * 0.1;
    return [min - padding, max + padding];
  }, [historicalData, forecastData]);

  // Find the transition point (where forecast starts)
  const transitionIndex = historicalData.length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="relative w-full"
    >
      {/* Glassmorphism Container */}
      <div className="backdrop-blur-xl bg-slate-900/60 border border-slate-700/50
                      rounded-2xl p-6 shadow-2xl shadow-purple-500/5">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <span className="text-2xl">{stockCode}</span>
              <span className="text-slate-400 text-sm font-normal">
                {stockName}
              </span>
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Probabilistic Forecast with Uncertainty Bands
            </p>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-gradient-to-r from-blue-500 to-cyan-400" />
              <span className="text-slate-400">Historical</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-gradient-to-b from-purple-500/60 to-indigo-500/20" />
              <span className="text-slate-400">Forecast Range</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 border-t-2 border-dashed border-cyan-400" />
              <span className="text-slate-400">Median</span>
            </div>
          </div>
        </div>

        {/* Chart */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <ResponsiveContainer width="100%" height={height}>
            <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <GradientDefs />

              {showGrid && (
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#334155"
                  strokeOpacity={0.3}
                  vertical={false}
                />
              )}

              <XAxis
                dataKey="date"
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickMargin={10}
              />

              <YAxis
                domain={yDomain}
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickFormatter={(value) => `${currency} ${value.toFixed(0)}`}
                tickMargin={10}
                width={80}
              />

              <Tooltip
                content={<CustomTooltip currency={currency} />}
                cursor={{
                  stroke: '#6366f1',
                  strokeWidth: 1,
                  strokeDasharray: '5 5',
                }}
              />

              {/* Reference line at transition point */}
              {transitionIndex > 0 && (
                <ReferenceLine
                  x={chartData[transitionIndex - 1]?.date}
                  stroke="#8b5cf6"
                  strokeDasharray="5 5"
                  strokeOpacity={0.5}
                  label={{
                    value: 'Forecast →',
                    position: 'top',
                    fill: '#a855f7',
                    fontSize: 10,
                  }}
                />
              )}

              {/* Forecast uncertainty area (range) */}
              <Area
                type="monotone"
                dataKey="range"
                fill="url(#uncertaintyGradient)"
                stroke="none"
                animationBegin={500}
                animationDuration={animationDuration * 1000}
                animationEasing="ease-out"
                filter="url(#areaGlow)"
              />

              {/* Historical price line */}
              <Line
                type="monotone"
                dataKey="price"
                stroke="url(#historicalGradient)"
                strokeWidth={2.5}
                dot={false}
                activeDot={{
                  r: 6,
                  fill: '#3b82f6',
                  stroke: '#fff',
                  strokeWidth: 2,
                }}
                animationDuration={animationDuration * 500}
              />

              {/* Forecast median line (dashed neon) */}
              <Line
                type="monotone"
                dataKey="median"
                stroke="#22d3ee"
                strokeWidth={2}
                strokeDasharray="8 4"
                dot={false}
                activeDot={{
                  r: 6,
                  fill: '#22d3ee',
                  stroke: '#fff',
                  strokeWidth: 2,
                }}
                filter="url(#neonGlow)"
                animationBegin={800}
                animationDuration={animationDuration * 1000}
                animationEasing="ease-out"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Forecast Stats */}
        {forecastData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1, duration: 0.5 }}
            className="mt-4 pt-4 border-t border-slate-700/50"
          >
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">5-Day Target</p>
                <p className="text-lg font-bold text-cyan-400">
                  {currency} {forecastData[forecastData.length - 1]?.median?.toFixed(2)}
                </p>
              </div>
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">Lower Bound (10%)</p>
                <p className="text-lg font-bold text-red-400">
                  {currency} {forecastData[forecastData.length - 1]?.lower?.toFixed(2)}
                </p>
              </div>
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">Upper Bound (90%)</p>
                <p className="text-lg font-bold text-green-400">
                  {currency} {forecastData[forecastData.length - 1]?.upper?.toFixed(2)}
                </p>
              </div>
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">Uncertainty Width</p>
                <p className="text-lg font-bold text-purple-400">
                  {((forecastData[forecastData.length - 1]?.upper -
                     forecastData[forecastData.length - 1]?.lower) /
                     forecastData[forecastData.length - 1]?.median * 100
                  ).toFixed(1)}%
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default UncertaintyChart;
