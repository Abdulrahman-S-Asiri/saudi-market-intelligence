/**
 * usePositions Hook - Position data management
 * Handles CRUD operations for position manager
 */

import { useState, useEffect, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

const usePositions = (autoRefresh = true, refreshInterval = 30000) => {
  const [positions, setPositions] = useState([]);
  const [summary, setSummary] = useState({
    total_positions: 0,
    open_positions: 0,
    closed_positions: 0,
    total_pnl: 0,
    unrealized_pnl: 0,
    wins: 0,
    losses: 0,
    win_rate: 0
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState(null); // null = ALL, 'OPEN', 'CLOSED'

  // Fetch all positions
  const fetchPositions = useCallback(async (statusFilter = filter) => {
    setLoading(true);
    setError(null);

    try {
      const url = statusFilter
        ? `${API_BASE}/api/positions?status=${statusFilter}`
        : `${API_BASE}/api/positions`;

      const response = await fetch(url);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch positions');
      }

      setPositions(data.positions || []);
      setSummary(data.summary || {});
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  // Create a new position
  const createPosition = useCallback(async (positionData) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/positions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(positionData)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to create position');
      }

      // Refresh positions list
      await fetchPositions();
      return data.position;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchPositions]);

  // Create position from signal
  const createFromSignal = useCallback(async (symbol, price, signalId = null) => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({ price: price.toString() });
      if (signalId) params.append('signal_id', signalId);

      const response = await fetch(
        `${API_BASE}/api/positions/from-signal/${symbol}?${params.toString()}`,
        { method: 'POST' }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to create position from signal');
      }

      // Refresh positions list
      await fetchPositions();
      return data.position;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchPositions]);

  // Close a position
  const closePosition = useCallback(async (positionId, exitPrice) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/positions/${positionId}/close`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exit_price: exitPrice })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to close position');
      }

      // Refresh positions list
      await fetchPositions();
      return data.position;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchPositions]);

  // Update a position
  const updatePosition = useCallback(async (positionId, updateData) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/positions/${positionId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updateData)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to update position');
      }

      // Refresh positions list
      await fetchPositions();
      return data.position;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchPositions]);

  // Delete a position
  const deletePosition = useCallback(async (positionId) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/positions/${positionId}`, {
        method: 'DELETE'
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to delete position');
      }

      // Refresh positions list
      await fetchPositions();
      return true;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchPositions]);

  // Update prices for a symbol
  const updatePricesForSymbol = useCallback(async (symbol, currentPrice) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/positions/update-prices/${symbol}?current_price=${currentPrice}`,
        { method: 'PUT' }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to update prices');
      }

      // Refresh positions list
      await fetchPositions();
      return data.updated_count;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  }, [fetchPositions]);

  // Initial fetch and auto-refresh
  useEffect(() => {
    fetchPositions();

    if (autoRefresh && refreshInterval > 0) {
      const interval = setInterval(() => {
        fetchPositions();
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [fetchPositions, autoRefresh, refreshInterval]);

  // Change filter
  const changeFilter = useCallback((newFilter) => {
    setFilter(newFilter);
    fetchPositions(newFilter);
  }, [fetchPositions]);

  return {
    positions,
    summary,
    loading,
    error,
    filter,
    changeFilter,
    fetchPositions,
    createPosition,
    createFromSignal,
    closePosition,
    updatePosition,
    deletePosition,
    updatePricesForSymbol
  };
};

export default usePositions;
