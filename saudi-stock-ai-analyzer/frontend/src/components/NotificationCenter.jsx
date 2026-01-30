// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

/**
 * NotificationCenter Component
 *
 * Slide-in notification system for signal alerts and updates.
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './NotificationCenter.css';

const NotificationCenter = ({ notifications = [], onDismiss, onClearAll }) => {
  const [isOpen, setIsOpen] = useState(false);
  const unreadCount = notifications.filter(n => !n.read).length;

  const getNotificationIcon = (type) => {
    switch (type) {
      case 'buy':
        return '📈';
      case 'sell':
        return '📉';
      case 'alert':
        return '⚠️';
      case 'success':
        return '✅';
      case 'error':
        return '❌';
      default:
        return '🔔';
    }
  };

  const getNotificationClass = (type) => {
    switch (type) {
      case 'buy':
        return 'notification-buy';
      case 'sell':
        return 'notification-sell';
      case 'alert':
        return 'notification-alert';
      case 'success':
        return 'notification-success';
      case 'error':
        return 'notification-error';
      default:
        return 'notification-default';
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="notification-center">
      {/* Notification Bell */}
      <button
        className="notification-bell"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Notifications"
      >
        <svg className="bell-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
          />
        </svg>
        {unreadCount > 0 && (
          <span className="notification-badge">{unreadCount}</span>
        )}
      </button>

      {/* Notification Panel */}
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              className="notification-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              className="notification-panel"
              initial={{ opacity: 0, x: 20, scale: 0.95 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 20, scale: 0.95 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            >
              <div className="notification-header">
                <h3 className="notification-title">Notifications</h3>
                {notifications.length > 0 && (
                  <button className="clear-all-btn" onClick={onClearAll}>
                    Clear all
                  </button>
                )}
              </div>

              <div className="notification-list">
                {notifications.length === 0 ? (
                  <div className="notification-empty">
                    <span className="empty-icon">🔔</span>
                    <span>No notifications</span>
                  </div>
                ) : (
                  notifications.map((notification, index) => (
                    <motion.div
                      key={notification.id || index}
                      className={`notification-item ${getNotificationClass(notification.type)} ${notification.read ? 'read' : ''}`}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <span className="notification-icon">
                        {getNotificationIcon(notification.type)}
                      </span>
                      <div className="notification-content">
                        <p className="notification-message">{notification.message}</p>
                        <span className="notification-time">
                          {formatTime(notification.timestamp)}
                        </span>
                      </div>
                      <button
                        className="notification-dismiss"
                        onClick={() => onDismiss?.(notification.id)}
                        aria-label="Dismiss"
                      >
                        ×
                      </button>
                    </motion.div>
                  ))
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

// Toast Notification Component
export const Toast = ({ message, type, onClose, duration = 5000 }) => {
  useEffect(() => {
    if (duration) {
      const timer = setTimeout(onClose, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, onClose]);

  return (
    <motion.div
      className={`toast toast-${type}`}
      initial={{ opacity: 0, y: 50, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 20, scale: 0.9 }}
      transition={{ type: 'spring', damping: 25 }}
    >
      <span className="toast-icon">
        {type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'buy' ? '📈' : type === 'sell' ? '📉' : '🔔'}
      </span>
      <span className="toast-message">{message}</span>
      <button className="toast-close" onClick={onClose}>×</button>
    </motion.div>
  );
};

export default NotificationCenter;
