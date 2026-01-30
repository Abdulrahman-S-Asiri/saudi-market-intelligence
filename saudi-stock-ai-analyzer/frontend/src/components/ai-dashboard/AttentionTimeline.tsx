/**
 * AttentionTimeline Component
 * ===========================
 * Visualizes AI attention weights as an interpretable heat strip.
 *
 * Features:
 * - Horizontal heat strip showing temporal attention
 * - High attention = Bright Neon Purple
 * - Low attention = Dark/Transparent
 * - Interactive tooltips explaining AI focus
 *
 * @author Claude AI / Abdulrahman Asiri
 * @version 1.0.0
 */

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// ============================================================================
// TYPES
// ============================================================================

interface AttentionDataPoint {
  date: string;
  weight: number;        // 0-1 normalized attention weight
  dayIndex?: number;     // Optional: day offset from current
  event?: string;        // Optional: notable event on this day
}

interface AttentionTimelineProps {
  data: AttentionDataPoint[];
  height?: number;
  showLabels?: boolean;
  colorScheme?: 'purple' | 'cyan' | 'emerald';
  title?: string;
}

// ============================================================================
// COLOR SCHEMES
// ============================================================================

const colorSchemes = {
  purple: {
    high: 'from-purple-500 via-fuchsia-500 to-pink-500',
    glow: 'shadow-purple-500/50',
    text: 'text-purple-400',
    bg: 'bg-purple-500',
    border: 'border-purple-500/30',
  },
  cyan: {
    high: 'from-cyan-500 via-blue-500 to-indigo-500',
    glow: 'shadow-cyan-500/50',
    text: 'text-cyan-400',
    bg: 'bg-cyan-500',
    border: 'border-cyan-500/30',
  },
  emerald: {
    high: 'from-emerald-500 via-teal-500 to-cyan-500',
    glow: 'shadow-emerald-500/50',
    text: 'text-emerald-400',
    bg: 'bg-emerald-500',
    border: 'border-emerald-500/30',
  },
};

// ============================================================================
// TOOLTIP COMPONENT
// ============================================================================

interface TooltipProps {
  data: AttentionDataPoint;
  position: { x: number; y: number };
  colorScheme: keyof typeof colorSchemes;
}

const AttentionTooltip: React.FC<TooltipProps> = ({ data, position, colorScheme }) => {
  const colors = colorSchemes[colorScheme];
  const attentionLevel = data.weight > 0.7 ? 'High' : data.weight > 0.4 ? 'Medium' : 'Low';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 5, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      className="absolute z-50 pointer-events-none"
      style={{
        left: `${position.x}px`,
        top: `${position.y - 120}px`,
        transform: 'translateX(-50%)',
      }}
    >
      <div className={`backdrop-blur-xl bg-slate-900/90 border ${colors.border}
                       rounded-xl p-4 shadow-2xl min-w-[220px]`}>
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-slate-400">{data.date}</span>
          <div className={`px-2 py-0.5 rounded-full text-xs font-medium
                          ${data.weight > 0.7 ? 'bg-purple-500/20 text-purple-300' :
                            data.weight > 0.4 ? 'bg-yellow-500/20 text-yellow-300' :
                            'bg-slate-500/20 text-slate-400'}`}>
            {attentionLevel} Focus
          </div>
        </div>

        {/* Attention Bar */}
        <div className="mb-3">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-slate-400">Attention Weight</span>
            <span className={colors.text}>{(data.weight * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${data.weight * 100}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
              className={`h-full bg-gradient-to-r ${colors.high} rounded-full`}
            />
          </div>
        </div>

        {/* AI Interpretation */}
        <div className="pt-3 border-t border-slate-700/50">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-purple-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <p className="text-xs text-slate-300 leading-relaxed">
              {data.weight > 0.7 ? (
                <>The AI <span className="text-purple-400 font-medium">heavily relied</span> on market patterns from this date for its prediction.</>
              ) : data.weight > 0.4 ? (
                <>This date provided <span className="text-yellow-400 font-medium">moderate influence</span> on the AI's forecast.</>
              ) : (
                <>This date had <span className="text-slate-400 font-medium">minimal impact</span> on the prediction.</>
              )}
            </p>
          </div>
        </div>

        {/* Event Badge (if any) */}
        {data.event && (
          <div className="mt-3 flex items-center gap-2 bg-slate-800/50 rounded-lg px-3 py-2">
            <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-xs text-amber-300">{data.event}</span>
          </div>
        )}

        {/* Arrow */}
        <div className="absolute left-1/2 -bottom-2 transform -translate-x-1/2">
          <div className={`w-4 h-4 rotate-45 bg-slate-900 border-r border-b ${colors.border}`} />
        </div>
      </div>
    </motion.div>
  );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const AttentionTimeline: React.FC<AttentionTimelineProps> = ({
  data,
  height = 48,
  showLabels = true,
  colorScheme = 'purple',
  title = 'AI Attention Timeline',
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const colors = colorSchemes[colorScheme];

  // Find max attention for highlighting
  const maxAttention = useMemo(() => {
    return Math.max(...data.map(d => d.weight));
  }, [data]);

  // Get color intensity based on attention weight
  const getBarStyle = (weight: number): React.CSSProperties => {
    const opacity = Math.max(0.1, weight);
    const brightness = weight > 0.7 ? 1.2 : 1;

    return {
      opacity,
      filter: weight > 0.5 ? `brightness(${brightness})` : 'none',
    };
  };

  // Handle mouse events
  const handleMouseEnter = (index: number, event: React.MouseEvent) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setHoveredIndex(index);
    setTooltipPosition({
      x: rect.left + rect.width / 2,
      y: rect.top,
    });
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
  };

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
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg bg-gradient-to-br ${colors.high} bg-opacity-20`}>
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">{title}</h3>
              <p className="text-xs text-slate-500">
                Hover over segments to see AI interpretation
              </p>
            </div>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded bg-gradient-to-r ${colors.high}`} />
              <span className="text-slate-400">High Attention</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-slate-700" />
              <span className="text-slate-400">Low Attention</span>
            </div>
          </div>
        </div>

        {/* Timeline Container */}
        <div className="relative">
          {/* Heat Strip */}
          <div
            className="flex gap-0.5 rounded-lg overflow-hidden"
            style={{ height: `${height}px` }}
          >
            {data.map((point, index) => (
              <motion.div
                key={point.date}
                initial={{ scaleY: 0, opacity: 0 }}
                animate={{ scaleY: 1, opacity: 1 }}
                transition={{
                  duration: 0.5,
                  delay: index * 0.01,
                  ease: 'easeOut',
                }}
                className={`flex-1 relative cursor-pointer transition-all duration-200
                           ${hoveredIndex === index ? 'ring-2 ring-white/50 z-10' : ''}`}
                style={{
                  ...getBarStyle(point.weight),
                  transformOrigin: 'bottom',
                }}
                onMouseEnter={(e) => handleMouseEnter(index, e)}
                onMouseLeave={handleMouseLeave}
              >
                {/* Gradient bar */}
                <div
                  className={`absolute inset-0 bg-gradient-to-t ${colors.high}
                             ${point.weight > 0.7 ? colors.glow + ' shadow-lg' : ''}`}
                  style={{
                    opacity: point.weight,
                  }}
                />

                {/* Peak indicator */}
                {point.weight === maxAttention && point.weight > 0.5 && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.5, type: 'spring' }}
                    className="absolute -top-1 left-1/2 transform -translate-x-1/2"
                  >
                    <div className="w-2 h-2 rounded-full bg-white shadow-lg shadow-white/50" />
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Date Labels */}
          {showLabels && (
            <div className="flex justify-between mt-2 px-1">
              <span className="text-xs text-slate-500">
                {data[0]?.date}
              </span>
              <span className="text-xs text-slate-400">
                60-Day Lookback
              </span>
              <span className="text-xs text-slate-500">
                {data[data.length - 1]?.date}
              </span>
            </div>
          )}
        </div>

        {/* Summary Stats */}
        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-xs text-slate-500 mb-1">Peak Attention</p>
              <p className={`text-lg font-bold ${colors.text}`}>
                {(maxAttention * 100).toFixed(0)}%
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-slate-500 mb-1">High Focus Days</p>
              <p className="text-lg font-bold text-white">
                {data.filter(d => d.weight > 0.5).length}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-slate-500 mb-1">Avg Attention</p>
              <p className="text-lg font-bold text-slate-300">
                {((data.reduce((acc, d) => acc + d.weight, 0) / data.length) * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Tooltip */}
      <AnimatePresence>
        {hoveredIndex !== null && data[hoveredIndex] && (
          <AttentionTooltip
            data={data[hoveredIndex]}
            position={tooltipPosition}
            colorScheme={colorScheme}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default AttentionTimeline;
