/**
 * RiskAdjustedBadge Component
 * ===========================
 * Smart badge for stock cards showing AI-generated signals.
 *
 * Signal Logic:
 * - High Return + High Confidence = Prime Opportunity (Green Glow)
 * - High Return + Low Confidence = Speculative Play (Yellow/Orange)
 * - Negative Trend = Bearish (Red)
 *
 * @author Claude AI / Abdulrahman Asiri
 * @version 1.0.0
 */

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

// ============================================================================
// TYPES
// ============================================================================

interface RiskAdjustedBadgeProps {
  predictedReturn: number;      // Expected return as decimal (0.02 = 2%)
  confidenceScore: number;      // 0-1 confidence score
  size?: 'sm' | 'md' | 'lg';
  showDetails?: boolean;
  animated?: boolean;
  className?: string;
}

type SignalType = 'prime' | 'bullish' | 'speculative' | 'neutral' | 'cautious' | 'bearish';

interface SignalConfig {
  type: SignalType;
  label: string;
  icon: string;
  description: string;
  bgClass: string;
  textClass: string;
  glowClass: string;
  borderClass: string;
  iconBg: string;
}

// ============================================================================
// SIGNAL CONFIGURATION
// ============================================================================

const signalConfigs: Record<SignalType, SignalConfig> = {
  prime: {
    type: 'prime',
    label: 'Prime Opportunity',
    icon: 'rocket',
    description: 'Strong signal with high confidence',
    bgClass: 'bg-gradient-to-r from-emerald-500/20 via-green-500/20 to-teal-500/20',
    textClass: 'text-emerald-400',
    glowClass: 'shadow-lg shadow-emerald-500/30',
    borderClass: 'border-emerald-500/50',
    iconBg: 'bg-emerald-500/20',
  },
  bullish: {
    type: 'bullish',
    label: 'Bullish Signal',
    icon: 'trending-up',
    description: 'Positive outlook with moderate confidence',
    bgClass: 'bg-gradient-to-r from-green-500/15 to-emerald-500/15',
    textClass: 'text-green-400',
    glowClass: 'shadow-md shadow-green-500/20',
    borderClass: 'border-green-500/40',
    iconBg: 'bg-green-500/20',
  },
  speculative: {
    type: 'speculative',
    label: 'Speculative Play',
    icon: 'zap',
    description: 'High potential but uncertain',
    bgClass: 'bg-gradient-to-r from-amber-500/20 via-orange-500/20 to-yellow-500/20',
    textClass: 'text-amber-400',
    glowClass: 'shadow-md shadow-amber-500/20',
    borderClass: 'border-amber-500/50 border-dashed',
    iconBg: 'bg-amber-500/20',
  },
  neutral: {
    type: 'neutral',
    label: 'Hold Position',
    icon: 'minus',
    description: 'No strong directional signal',
    bgClass: 'bg-slate-500/10',
    textClass: 'text-slate-400',
    glowClass: '',
    borderClass: 'border-slate-500/30',
    iconBg: 'bg-slate-500/20',
  },
  cautious: {
    type: 'cautious',
    label: 'Exercise Caution',
    icon: 'alert-triangle',
    description: 'Mixed signals detected',
    bgClass: 'bg-gradient-to-r from-yellow-500/15 to-orange-500/15',
    textClass: 'text-yellow-400',
    glowClass: '',
    borderClass: 'border-yellow-500/40',
    iconBg: 'bg-yellow-500/20',
  },
  bearish: {
    type: 'bearish',
    label: 'Bearish Outlook',
    icon: 'trending-down',
    description: 'Negative momentum detected',
    bgClass: 'bg-gradient-to-r from-red-500/20 via-rose-500/20 to-pink-500/20',
    textClass: 'text-red-400',
    glowClass: 'shadow-md shadow-red-500/20',
    borderClass: 'border-red-500/50',
    iconBg: 'bg-red-500/20',
  },
};

// ============================================================================
// ICONS
// ============================================================================

const Icons: Record<string, React.FC<{ className?: string }>> = {
  rocket: ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  'trending-up': ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  ),
  'trending-down': ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
    </svg>
  ),
  zap: ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  minus: ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
    </svg>
  ),
  'alert-triangle': ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
};

// ============================================================================
// SIZE CONFIGURATIONS
// ============================================================================

const sizeConfigs = {
  sm: {
    container: 'px-3 py-1.5 text-xs',
    icon: 'w-3.5 h-3.5',
    iconContainer: 'p-1',
    gap: 'gap-1.5',
  },
  md: {
    container: 'px-4 py-2 text-sm',
    icon: 'w-4 h-4',
    iconContainer: 'p-1.5',
    gap: 'gap-2',
  },
  lg: {
    container: 'px-5 py-3 text-base',
    icon: 'w-5 h-5',
    iconContainer: 'p-2',
    gap: 'gap-3',
  },
};

// ============================================================================
// SIGNAL DETERMINATION LOGIC
// ============================================================================

function determineSignal(predictedReturn: number, confidenceScore: number): SignalConfig {
  // Thresholds matching backend logic
  const STRONG_RETURN = 0.015;  // 1.5%
  const POSITIVE_RETURN = 0.01; // 1.0%
  const NEGATIVE_RETURN = -0.005; // -0.5%
  const HIGH_CONFIDENCE = 0.7;
  const MEDIUM_CONFIDENCE = 0.4;

  // Prime Opportunity: High return + High confidence
  if (predictedReturn > STRONG_RETURN && confidenceScore >= HIGH_CONFIDENCE) {
    return signalConfigs.prime;
  }

  // Bullish: Positive return + Medium-High confidence
  if (predictedReturn > POSITIVE_RETURN && confidenceScore >= MEDIUM_CONFIDENCE) {
    return signalConfigs.bullish;
  }

  // Speculative: High potential but low confidence
  if (predictedReturn > STRONG_RETURN && confidenceScore < MEDIUM_CONFIDENCE) {
    return signalConfigs.speculative;
  }

  // Bearish: Strong negative signal
  if (predictedReturn < NEGATIVE_RETURN) {
    if (confidenceScore >= HIGH_CONFIDENCE) {
      return signalConfigs.bearish;
    }
    return signalConfigs.cautious;
  }

  // Neutral: Everything else
  return signalConfigs.neutral;
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const RiskAdjustedBadge: React.FC<RiskAdjustedBadgeProps> = ({
  predictedReturn,
  confidenceScore,
  size = 'md',
  showDetails = false,
  animated = true,
  className = '',
}) => {
  // Determine signal type
  const signal = useMemo(
    () => determineSignal(predictedReturn, confidenceScore),
    [predictedReturn, confidenceScore]
  );

  const sizeConfig = sizeConfigs[size];
  const IconComponent = Icons[signal.icon];

  // Animation variants
  const containerVariants = {
    initial: { opacity: 0, scale: 0.9 },
    animate: {
      opacity: 1,
      scale: 1,
      transition: { duration: 0.3, ease: 'easeOut' },
    },
    hover: {
      scale: 1.02,
      transition: { duration: 0.2 },
    },
  };

  const pulseVariants = {
    animate: {
      scale: [1, 1.05, 1],
      opacity: [0.5, 0.8, 0.5],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial={animated ? 'initial' : false}
      animate="animate"
      whileHover="hover"
      className={`relative inline-flex ${className}`}
    >
      {/* Glow effect for prime/bullish signals */}
      {signal.type === 'prime' && animated && (
        <motion.div
          variants={pulseVariants}
          animate="animate"
          className={`absolute inset-0 rounded-xl ${signal.bgClass} blur-xl`}
        />
      )}

      {/* Main Badge */}
      <div
        className={`
          relative backdrop-blur-xl ${signal.bgClass}
          border ${signal.borderClass} ${signal.glowClass}
          rounded-xl ${sizeConfig.container} ${sizeConfig.gap}
          flex items-center font-medium
          transition-all duration-300
        `}
      >
        {/* Icon */}
        <div className={`${signal.iconBg} rounded-lg ${sizeConfig.iconContainer}`}>
          <IconComponent className={`${sizeConfig.icon} ${signal.textClass}`} />
        </div>

        {/* Label */}
        <span className={signal.textClass}>{signal.label}</span>

        {/* Return Badge */}
        <div className={`
          ml-1 px-2 py-0.5 rounded-md text-xs font-bold
          ${predictedReturn >= 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}
        `}>
          {predictedReturn >= 0 ? '+' : ''}{(predictedReturn * 100).toFixed(1)}%
        </div>

        {/* Confidence Indicator (for detailed view) */}
        {showDetails && (
          <div className="flex items-center gap-1 ml-2 pl-2 border-l border-slate-600/30">
            <div className="flex gap-0.5">
              {[0.25, 0.5, 0.75, 1].map((threshold) => (
                <div
                  key={threshold}
                  className={`w-1 h-3 rounded-full ${
                    confidenceScore >= threshold
                      ? signal.type === 'prime' || signal.type === 'bullish'
                        ? 'bg-green-400'
                        : signal.type === 'bearish'
                        ? 'bg-red-400'
                        : 'bg-amber-400'
                      : 'bg-slate-600'
                  }`}
                />
              ))}
            </div>
            <span className="text-xs text-slate-400 ml-1">
              {(confidenceScore * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

// ============================================================================
// EXTENDED CARD COMPONENT
// ============================================================================

interface RiskAdjustedCardProps extends RiskAdjustedBadgeProps {
  stockCode: string;
  stockName: string;
  currentPrice: number;
  currency?: string;
}

export const RiskAdjustedCard: React.FC<RiskAdjustedCardProps> = ({
  stockCode,
  stockName,
  currentPrice,
  currency = 'SAR',
  predictedReturn,
  confidenceScore,
  animated = true,
}) => {
  const signal = useMemo(
    () => determineSignal(predictedReturn, confidenceScore),
    [predictedReturn, confidenceScore]
  );

  const targetPrice = currentPrice * (1 + predictedReturn);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className={`
        relative backdrop-blur-xl bg-slate-900/60
        border border-slate-700/50 ${signal.glowClass}
        rounded-2xl p-5 shadow-2xl
        transition-all duration-300
        hover:border-slate-600/70
      `}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-xl font-bold text-white">{stockCode}</h3>
          <p className="text-sm text-slate-400">{stockName}</p>
        </div>
        <RiskAdjustedBadge
          predictedReturn={predictedReturn}
          confidenceScore={confidenceScore}
          size="sm"
          animated={animated}
        />
      </div>

      {/* Price Section */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-xs text-slate-500 mb-1">Current Price</p>
          <p className="text-lg font-semibold text-white">
            {currency} {currentPrice.toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">5-Day Target</p>
          <p className={`text-lg font-semibold ${signal.textClass}`}>
            {currency} {targetPrice.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="mb-4">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-slate-500">AI Confidence</span>
          <span className={signal.textClass}>{(confidenceScore * 100).toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidenceScore * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut', delay: 0.3 }}
            className={`h-full rounded-full ${
              signal.type === 'prime' || signal.type === 'bullish'
                ? 'bg-gradient-to-r from-green-500 to-emerald-400'
                : signal.type === 'bearish'
                ? 'bg-gradient-to-r from-red-500 to-rose-400'
                : 'bg-gradient-to-r from-amber-500 to-yellow-400'
            }`}
          />
        </div>
      </div>

      {/* Signal Description */}
      <div className={`
        flex items-center gap-2 p-3 rounded-lg
        ${signal.bgClass} border ${signal.borderClass}
      `}>
        <svg className={`w-4 h-4 ${signal.textClass}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-xs text-slate-300">{signal.description}</p>
      </div>
    </motion.div>
  );
};

export default RiskAdjustedBadge;
