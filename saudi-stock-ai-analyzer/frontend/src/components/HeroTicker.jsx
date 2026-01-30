// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

/**
 * HeroTicker Component
 *
 * Animated live price display with signal badge and particle effects.
 * The main focal point showing current price, change, and trading signal.
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './HeroTicker.css';

const HeroTicker = ({
  symbol,
  name,
  price,
  previousPrice,
  change,
  changePercent,
  signal,
  confidence,
  isLoading
}) => {
  const [priceFlash, setPriceFlash] = useState(null);
  const [particles, setParticles] = useState([]);
  const prevPriceRef = useRef(price);

  // Animate price changes
  useEffect(() => {
    if (price !== prevPriceRef.current && !isLoading) {
      const direction = price > prevPriceRef.current ? 'up' : 'down';
      setPriceFlash(direction);

      // Create particles on significant changes
      if (Math.abs(changePercent) > 0.5) {
        createParticles(direction);
      }

      const timeout = setTimeout(() => setPriceFlash(null), 500);
      prevPriceRef.current = price;
      return () => clearTimeout(timeout);
    }
  }, [price, changePercent, isLoading]);

  const createParticles = (direction) => {
    const newParticles = Array.from({ length: 8 }, (_, i) => ({
      id: Date.now() + i,
      x: Math.random() * 100 - 50,
      y: direction === 'up' ? -50 - Math.random() * 30 : 50 + Math.random() * 30,
      delay: i * 0.05
    }));
    setParticles(newParticles);
    setTimeout(() => setParticles([]), 1000);
  };

  const formatPrice = (p) => {
    if (!p) return '---';
    return new Intl.NumberFormat('en-SA', {
      style: 'decimal',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(p);
  };

  const getSignalColor = () => {
    if (!signal) return 'neutral';
    switch (signal.toLowerCase()) {
      case 'buy':
      case 'strong buy':
        return 'buy';
      case 'sell':
      case 'strong sell':
        return 'sell';
      default:
        return 'neutral';
    }
  };

  const signalColor = getSignalColor();
  const isPositive = change > 0;
  const changeClass = isPositive ? 'positive' : change < 0 ? 'negative' : 'neutral';

  return (
    <motion.div
      className="hero-ticker"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Background Glow */}
      <div className={`hero-glow hero-glow-${signalColor}`} />

      {/* Stock Info */}
      <div className="hero-stock-info">
        <span className="hero-symbol">{symbol}</span>
        <span className="hero-name">{name}</span>
      </div>

      {/* Main Price Display */}
      <div className="hero-price-container">
        {/* Particles */}
        <AnimatePresence>
          {particles.map((particle) => (
            <motion.div
              key={particle.id}
              className={`price-particle ${changeClass}`}
              initial={{ opacity: 1, x: 0, y: 0 }}
              animate={{ opacity: 0, x: particle.x, y: particle.y }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.8, delay: particle.delay }}
            />
          ))}
        </AnimatePresence>

        {/* Price */}
        {isLoading ? (
          <div className="hero-price-skeleton" />
        ) : (
          <motion.div
            className={`hero-price ${priceFlash ? `flash-${priceFlash}` : ''}`}
            key={price}
            initial={{ scale: 1.05, opacity: 0.8 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <span className="price-currency">SAR</span>
            <span className="price-value">{formatPrice(price)}</span>
          </motion.div>
        )}

        {/* Change */}
        <div className={`hero-change ${changeClass}`}>
          <span className="change-icon">
            {isPositive ? '▲' : change < 0 ? '▼' : '−'}
          </span>
          <span className="change-value">
            {isPositive ? '+' : ''}{formatPrice(change)}
          </span>
          <span className="change-percent">
            ({isPositive ? '+' : ''}{changePercent?.toFixed(2)}%)
          </span>
        </div>
      </div>

      {/* Signal Badge */}
      <AnimatePresence mode="wait">
        {signal && (
          <motion.div
            className={`hero-signal signal-${signalColor}`}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          >
            <div className="signal-pulse" />
            <span className="signal-label">{signal.toUpperCase()}</span>
            {confidence && (
              <span className="signal-confidence">{confidence}%</span>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Live Indicator */}
      <div className="hero-live-indicator">
        <span className="live-dot" />
        <span className="live-text">LIVE</span>
      </div>
    </motion.div>
  );
};

export default HeroTicker;
