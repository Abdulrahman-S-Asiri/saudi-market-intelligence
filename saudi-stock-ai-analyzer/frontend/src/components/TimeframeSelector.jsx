// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

import React from 'react';
import { motion } from 'framer-motion';

const timeframes = [
  { value: '1mo', label: '1M' },
  { value: '3mo', label: '3M' },
  { value: '6mo', label: '6M' },
  { value: '1y', label: '1Y' },
  { value: '2y', label: '2Y' },
];

const TimeframeSelector = ({ selected, onSelect }) => {
  return (
    <div className="timeframe-selector">
      {timeframes.map((tf) => (
        <motion.button
          key={tf.value}
          className={`timeframe-btn ${selected === tf.value ? 'active' : ''}`}
          onClick={() => onSelect(tf.value)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {tf.label}
          {selected === tf.value && (
            <motion.div
              className="active-indicator"
              layoutId="activeTimeframe"
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            />
          )}
        </motion.button>
      ))}
    </div>
  );
};

export default TimeframeSelector;
