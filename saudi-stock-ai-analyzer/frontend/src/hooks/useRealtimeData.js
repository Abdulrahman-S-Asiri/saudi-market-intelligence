// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

/**
 * useRealtimeData Hook
 *
 * Provides real-time stock data updates via WebSocket connection.
 * Falls back to polling if WebSocket is unavailable.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const RECONNECT_INTERVAL = 5000;
const POLLING_INTERVAL = 10000;
const MAX_RECONNECT_ATTEMPTS = 5;

export const ConnectionStatus = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  ERROR: 'error',
  POLLING: 'polling'
};

const useRealtimeData = (symbol, enabled = true) => {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState(ConnectionStatus.DISCONNECTED);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const wsRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef(null);
  const pollingTimer = useRef(null);

  // Fetch data via REST API (fallback)
  const fetchData = useCallback(async () => {
    if (!symbol) return;

    try {
      const response = await fetch(`${API_URL}/analyze/${symbol}?period=1mo`);
      if (!response.ok) throw new Error('Failed to fetch data');

      const result = await response.json();
      setData(result);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, [symbol]);

  // Start polling fallback
  const startPolling = useCallback(() => {
    if (pollingTimer.current) return;

    setStatus(ConnectionStatus.POLLING);
    fetchData();

    pollingTimer.current = setInterval(fetchData, POLLING_INTERVAL);
  }, [fetchData]);

  // Stop polling
  const stopPolling = useCallback(() => {
    if (pollingTimer.current) {
      clearInterval(pollingTimer.current);
      pollingTimer.current = null;
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!symbol || !enabled) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus(ConnectionStatus.CONNECTING);

    try {
      wsRef.current = new WebSocket(`${WS_URL}/${symbol}`);

      wsRef.current.onopen = () => {
        setStatus(ConnectionStatus.CONNECTED);
        setError(null);
        reconnectAttempts.current = 0;
        stopPolling();
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);

          if (message.type === 'price_update') {
            setData(prev => ({
              ...prev,
              signal: {
                ...prev?.signal,
                price: message.price
              },
              price_change: message.change,
              price_change_percent: message.change_percent
            }));
          } else if (message.type === 'analysis_update') {
            setData(message.data);
          } else if (message.type === 'signal_alert') {
            // Trigger notification
            if (window.dispatchEvent) {
              window.dispatchEvent(new CustomEvent('signal_alert', {
                detail: message
              }));
            }
          }

          setLastUpdate(new Date());
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      wsRef.current.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('Connection error');
      };

      wsRef.current.onclose = (event) => {
        setStatus(ConnectionStatus.DISCONNECTED);

        // Attempt reconnection
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts.current++;
          reconnectTimer.current = setTimeout(() => {
            connect();
          }, RECONNECT_INTERVAL * reconnectAttempts.current);
        } else {
          // Fall back to polling
          startPolling();
        }
      };
    } catch (err) {
      setStatus(ConnectionStatus.ERROR);
      setError(err.message);
      startPolling();
    }
  }, [symbol, enabled, startPolling, stopPolling]);

  // Disconnect WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    stopPolling();
    setStatus(ConnectionStatus.DISCONNECTED);
  }, [stopPolling]);

  // Send message to server
  const sendMessage = useCallback((message) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Subscribe to symbol changes
  const subscribe = useCallback((newSymbol) => {
    sendMessage({
      type: 'subscribe',
      symbol: newSymbol
    });
  }, [sendMessage]);

  // Unsubscribe from symbol
  const unsubscribe = useCallback((oldSymbol) => {
    sendMessage({
      type: 'unsubscribe',
      symbol: oldSymbol
    });
  }, [sendMessage]);

  // Effect: Connect/disconnect based on enabled state
  useEffect(() => {
    if (enabled && symbol) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, symbol, connect, disconnect]);

  // Effect: Handle symbol changes while connected
  useEffect(() => {
    if (status === ConnectionStatus.CONNECTED && symbol) {
      subscribe(symbol);
    }
  }, [symbol, status, subscribe]);

  // Manual refresh
  const refresh = useCallback(() => {
    if (status === ConnectionStatus.CONNECTED) {
      sendMessage({ type: 'refresh' });
    } else {
      fetchData();
    }
  }, [status, sendMessage, fetchData]);

  return {
    data,
    status,
    error,
    lastUpdate,
    isConnected: status === ConnectionStatus.CONNECTED,
    isPolling: status === ConnectionStatus.POLLING,
    connect,
    disconnect,
    refresh,
    subscribe,
    unsubscribe
  };
};

// Hook for managing multiple symbol subscriptions
export const useMultiSymbolData = (symbols = [], enabled = true) => {
  const [dataMap, setDataMap] = useState({});
  const [status, setStatus] = useState(ConnectionStatus.DISCONNECTED);

  const wsRef = useRef(null);

  const connect = useCallback(() => {
    if (!enabled || symbols.length === 0) return;

    try {
      wsRef.current = new WebSocket(`${WS_URL}/multi`);

      wsRef.current.onopen = () => {
        setStatus(ConnectionStatus.CONNECTED);

        // Subscribe to all symbols
        wsRef.current.send(JSON.stringify({
          type: 'subscribe_multi',
          symbols
        }));
      };

      wsRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.symbol) {
          setDataMap(prev => ({
            ...prev,
            [message.symbol]: message.data
          }));
        }
      };

      wsRef.current.onclose = () => {
        setStatus(ConnectionStatus.DISCONNECTED);
      };
    } catch (err) {
      setStatus(ConnectionStatus.ERROR);
    }
  }, [symbols, enabled]);

  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [enabled, connect]);

  return {
    dataMap,
    status,
    isConnected: status === ConnectionStatus.CONNECTED
  };
};

// Hook for signal alerts
export const useSignalAlerts = (onAlert) => {
  useEffect(() => {
    const handleAlert = (event) => {
      if (onAlert) {
        onAlert(event.detail);
      }
    };

    window.addEventListener('signal_alert', handleAlert);
    return () => window.removeEventListener('signal_alert', handleAlert);
  }, [onAlert]);
};

export default useRealtimeData;
