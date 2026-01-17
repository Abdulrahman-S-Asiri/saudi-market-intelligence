import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const SignalHistory = ({ signals, loading }) => {
  if (loading) {
    return (
      <div className="signal-history glass-card">
        <h3>Signal History</h3>
        <div className="signal-history-loading">
          <div className="loading-spinner small"></div>
          <span>Loading history...</span>
        </div>
      </div>
    );
  }

  if (!signals || signals.length === 0) {
    return (
      <div className="signal-history glass-card">
        <h3>Signal History</h3>
        <div className="signal-history-empty">
          <p>No signal history available</p>
        </div>
      </div>
    );
  }

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'BUY': return '#00c853';
      case 'SELL': return '#ff1744';
      default: return '#ffc107';
    }
  };

  const getSignalIcon = (signal) => {
    switch (signal) {
      case 'BUY': return '+';
      case 'SELL': return '-';
      default: return '=';
    }
  };

  return (
    <div className="signal-history glass-card">
      <div className="signal-history-header">
        <h3>Signal History</h3>
        <div className="signal-stats">
          <span className="stat-item buy">
            {signals.filter(s => s.signal === 'BUY').length} Buy
          </span>
          <span className="stat-item sell">
            {signals.filter(s => s.signal === 'SELL').length} Sell
          </span>
          <span className="stat-item hold">
            {signals.filter(s => s.signal === 'HOLD').length} Hold
          </span>
        </div>
      </div>

      <div className="signal-timeline">
        <AnimatePresence>
          {signals.slice(0, 10).map((item, index) => (
            <motion.div
              key={item.timestamp}
              className="timeline-item"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.05 }}
            >
              <div className="timeline-connector">
                <div
                  className="timeline-dot"
                  style={{ backgroundColor: getSignalColor(item.signal) }}
                >
                  {getSignalIcon(item.signal)}
                </div>
                {index < signals.length - 1 && <div className="timeline-line" />}
              </div>

              <div className="timeline-content">
                <div className="timeline-header">
                  <span
                    className="signal-badge small"
                    style={{ backgroundColor: getSignalColor(item.signal) }}
                  >
                    {item.signal}
                  </span>
                  <span className="timeline-date">
                    {new Date(item.timestamp).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                    })}
                  </span>
                </div>

                <div className="timeline-details">
                  <span className="price">
                    {item.price?.toFixed(2)} SAR
                  </span>
                  <span className="confidence">
                    {item.confidence?.toFixed(0)}% confidence
                  </span>
                </div>

                {item.outcome && (
                  <div
                    className={`timeline-outcome ${item.outcome.correct ? 'correct' : 'incorrect'}`}
                  >
                    {item.outcome.correct ? 'Correct' : 'Incorrect'}
                    {item.outcome.return_pct && (
                      <span className="return">
                        ({item.outcome.return_pct > 0 ? '+' : ''}
                        {item.outcome.return_pct.toFixed(2)}%)
                      </span>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {signals.length > 10 && (
        <div className="signal-history-more">
          <button className="view-more-btn">
            View all {signals.length} signals
          </button>
        </div>
      )}
    </div>
  );
};

export default SignalHistory;
