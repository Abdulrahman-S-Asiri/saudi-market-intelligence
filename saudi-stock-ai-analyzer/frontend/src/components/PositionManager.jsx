/**
 * PositionManager Component
 * Main dashboard for position tracking with summary and list view
 */

import React, { useState } from 'react';
import usePositions from '../hooks/usePositions';
import PositionCard from './PositionCard';
import AddPositionModal from './AddPositionModal';
import './PositionManager.css';

const PositionManager = ({ stocks }) => {
  const {
    positions,
    summary,
    loading,
    error,
    filter,
    changeFilter,
    createPosition,
    closePosition,
    deletePosition,
    fetchPositions
  } = usePositions(true, 30000); // Auto-refresh every 30 seconds

  const [showAddModal, setShowAddModal] = useState(false);

  const formatCurrency = (value) => {
    return value?.toFixed(2) || '0.00';
  };

  // Create stock info map for position cards
  const stockInfoMap = {};
  if (Array.isArray(stocks)) {
    stocks.forEach(stock => {
      stockInfoMap[stock.symbol] = stock;
    });
  }

  return (
    <div className="position-manager">
      {/* Header */}
      <div className="pm-header">
        <div className="pm-title">
          <h2>Position Manager</h2>
          <span className="pm-subtitle">Track your trades</span>
        </div>
        <button
          className="btn-add-position"
          onClick={() => setShowAddModal(true)}
        >
          + Add Position
        </button>
      </div>

      {/* Summary Bar */}
      <div className="pm-summary">
        <div className="summary-item">
          <span className="summary-label">Open</span>
          <span className="summary-value">{summary.open_positions}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Closed</span>
          <span className="summary-value">{summary.closed_positions}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Realized P&L</span>
          <span
            className="summary-value"
            style={{ color: summary.total_pnl >= 0 ? '#00c853' : '#ff1744' }}
          >
            {summary.total_pnl >= 0 ? '+' : ''}{formatCurrency(summary.total_pnl)} SAR
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Unrealized P&L</span>
          <span
            className="summary-value"
            style={{ color: summary.unrealized_pnl >= 0 ? '#00c853' : '#ff1744' }}
          >
            {summary.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(summary.unrealized_pnl)} SAR
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Win Rate</span>
          <span className="summary-value win-rate">
            {summary.win_rate?.toFixed(0) || 0}%
            <span className="win-loss-count">
              ({summary.wins}W / {summary.losses}L)
            </span>
          </span>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="pm-filters">
        <button
          className={`filter-tab ${filter === null ? 'active' : ''}`}
          onClick={() => changeFilter(null)}
        >
          All ({summary.total_positions})
        </button>
        <button
          className={`filter-tab ${filter === 'OPEN' ? 'active' : ''}`}
          onClick={() => changeFilter('OPEN')}
        >
          Open ({summary.open_positions})
        </button>
        <button
          className={`filter-tab ${filter === 'CLOSED' ? 'active' : ''}`}
          onClick={() => changeFilter('CLOSED')}
        >
          Closed ({summary.closed_positions})
        </button>
        <button
          className="btn-refresh"
          onClick={() => fetchPositions()}
          disabled={loading}
        >
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="pm-error">
          <span>Error: {error}</span>
          <button onClick={() => fetchPositions()}>Retry</button>
        </div>
      )}

      {/* Position List */}
      <div className="pm-positions">
        {loading && positions.length === 0 ? (
          <div className="pm-loading">
            <div className="spinner"></div>
            <span>Loading positions...</span>
          </div>
        ) : positions.length === 0 ? (
          <div className="pm-empty">
            <span className="empty-icon">📊</span>
            <h4>No positions yet</h4>
            <p>Add a position manually or create one from a BUY signal</p>
            <button
              className="btn-add-first"
              onClick={() => setShowAddModal(true)}
            >
              Add Your First Position
            </button>
          </div>
        ) : (
          <div className="positions-grid">
            {positions.map(position => (
              <PositionCard
                key={position.id}
                position={position}
                onClose={closePosition}
                onDelete={deletePosition}
                stockInfo={stockInfoMap[position.symbol]}
              />
            ))}
          </div>
        )}
      </div>

      {/* Add Position Modal */}
      <AddPositionModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onCreate={createPosition}
        stocks={Array.isArray(stocks) ? stocks : []}
      />
    </div>
  );
};

export default PositionManager;
