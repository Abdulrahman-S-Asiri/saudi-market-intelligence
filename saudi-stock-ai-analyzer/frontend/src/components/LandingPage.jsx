/**
 * Landing Page Component - TASI AI
 * Premium, luxurious welcome screen with Framer Motion animations
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './LandingPage.css';

const LandingPage = ({ onEnter }) => {
  return (
    <AnimatePresence>
      <motion.div
        className="landing-page"
        initial={{ opacity: 1 }}
        exit={{ opacity: 0, scale: 1.1 }}
        transition={{ duration: 0.6, ease: "easeInOut" }}
      >
        {/* Animated Background */}
        <div className="landing-bg">
          <div className="landing-gradient" />
          <div className="landing-grid" />
          <div className="landing-glow glow-1" />
          <div className="landing-glow glow-2" />
          <div className="landing-glow glow-3" />

          {/* Floating Particles */}
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="particle"
              initial={{
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
                opacity: 0
              }}
              animate={{
                y: [null, Math.random() * -200 - 100],
                opacity: [0, 0.6, 0]
              }}
              transition={{
                duration: Math.random() * 5 + 5,
                repeat: Infinity,
                delay: Math.random() * 5,
                ease: "linear"
              }}
            />
          ))}
        </div>

        {/* Content */}
        <div className="landing-content">
          {/* Logo Animation */}
          <motion.div
            className="landing-logo"
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{
              type: "spring",
              stiffness: 200,
              damping: 20,
              delay: 0.2
            }}
          >
            <div className="logo-ring ring-outer" />
            <div className="logo-ring ring-middle" />
            <div className="logo-ring ring-inner" />
            <span className="logo-icon-landing">📊</span>
          </motion.div>

          {/* Title */}
          <motion.h1
            className="landing-title"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            <span className="title-main">TASI</span>
            <span className="title-accent">AI</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            className="landing-subtitle"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
          >
            Institutional-Grade Saudi Market Analysis
          </motion.p>

          {/* Features */}
          <motion.div
            className="landing-features"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.9 }}
          >
            <div className="feature">
              <span className="feature-icon">🧠</span>
              <span>BiLSTM + Attention</span>
            </div>
            <div className="feature-divider" />
            <div className="feature">
              <span className="feature-icon">📈</span>
              <span>209+ Stocks</span>
            </div>
            <div className="feature-divider" />
            <div className="feature">
              <span className="feature-icon">⚡</span>
              <span>Real-time Signals</span>
            </div>
          </motion.div>

          {/* CTA Button */}
          <motion.button
            className="landing-cta"
            onClick={onEnter}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 1.1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="cta-text">Start Analysis</span>
            <span className="cta-icon">→</span>
            <div className="cta-glow" />
          </motion.button>

          {/* Version Badge */}
          <motion.div
            className="landing-version"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 1.3 }}
          >
            <span className="version-badge">v3.0</span>
            <span className="version-text">Advanced LSTM Edition</span>
          </motion.div>
        </div>

        {/* Bottom Decoration */}
        <motion.div
          className="landing-footer-decoration"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1.5 }}
        >
          <div className="decoration-line" />
          <span className="decoration-text">Powered by Deep Learning</span>
          <div className="decoration-line" />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default LandingPage;
