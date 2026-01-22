/**
 * PositionCard Component
 * Displays individual position with P&L and actions
 */

import React, { useState } from 'react';

const PositionCard = ({ position, onClose, onDelete, stockInfo }) => {
  const [showCloseModal, setShowCloseModal] = useState(false);
  const [exitPrice, setExitPrice] = useState(position.current_price || '');
  const [closing, setClosing] = useState(false);

  const getResultColor = (result) => {
    switch (result) {
      case 'WIN': return '#00c853';
      case 'LOSS': return '#ff1744';
      default: return '#ffc107';
    }
  };

  const getResultBg = (result) => {
    switch (result) {
      case 'WIN': return 'rgba(0, 200, 83, 0.15)';
      case 'LOSS': return 'rgba(255, 23, 68, 0.15)';
      default: return 'rgba(255, 193, 7, 0.15)';
    }
  };

  const formatPrice = (price) => {
    return price ? price.toFixed(2) : '-';
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const handleClose = async () => {
    if (!exitPrice || exitPrice <= 0) return;
    setClosing(true);
    try {
      await onClose(position.id, parseFloat(exitPrice));
      setShowCloseModal(false);
    } catch (err) {
      console.error('Failed to close position:', err);
    } finally {
      setClosing(false);
    }
  };

  const handleDelete = async () => {
    if (window.confirm('Are you sure you want to delete this position?')) {
      try {
        await onDelete(position.id);
      } catch (err) {
        console.error('Failed to delete position:', err);
      }
    }
  };

  const isOpen = position.status === 'OPEN';
  const stockName = stockInfo?.name || position.symbol;

  return (
    <div className="position-card">
      {/* Header */}
      <div className="position-header">
        <div className="position-symbol">
          <span className="symbol">{position.symbol}</span>
          <span className="name">{stockName}</span>
        </div>
        <div
          className="result-badge"
          style={{
            backgroundColor: getResultBg(position.result),
            color: getResultColor(position.result),
            borderColor: getResultColor(position.result)
          }}
        >
          {position.result}
        </div>
      </div>

      {/* Price Info */}
      <div className="position-prices">
        <div className="price-item">
          <span className="price-label">Entry</span>
          <span className="price-value">{formatPrice(position.entry_price)} SAR</span>
        </div>
        <div className="price-item">
          <span className="price-label">{isOpen ? 'Current' : 'Exit'}</span>
          <span className="price-value">
            {formatPrice(isOpen ? position.current_price : position.exit_price)} SAR
          </span>
        </div>
      </div>

      {/* P&L Display */}
      <div className="position-pnl">
        <div
          className="pnl-amount"
          style={{ color: position.pnl >= 0 ? '#00c853' : '#ff1744' }}
        >
          {position.pnl >= 0 ? '+' : ''}{formatPrice(position.pnl)} SAR
        </div>
        <div
          className="pnl-percentage"
          style={{ color: position.pnl_percentage >= 0 ? '#00c853' : '#ff1744' }}
        >
          {position.pnl_percentage >= 0 ? '+' : ''}{position.pnl_percentage?.toFixed(2)}%
        </div>
      </div>

      {/* Meta Info */}
      <div className="position-meta">
        <div className="meta-item">
          <span className="meta-label">Qty</span>
          <span className="meta-value">{position.quantity}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Source</span>
          <span className={`meta-value source-${position.source?.toLowerCase()}`}>
            {position.source}
          </span>
        </div>
        <div className="meta-item">
          <span className="meta-label">{isOpen ? 'Opened' : 'Closed'}</span>
          <span className="meta-value">
            {formatDate(isOpen ? position.entry_date : position.exit_date)}
          </span>
        </div>
      </div>

      {/* Notes */}
      {position.notes && (
        <div className="position-notes">
          <span className="notes-label">Notes:</span>
          <span className="notes-text">{position.notes}</span>
        </div>
      )}

      {/* Actions */}
      <div className="position-actions">
        {isOpen && (
          <button
            className="btn-close-position"
            onClick={() => setShowCloseModal(true)}
          >
            Close Position
          </button>
        )}
        <button
          className="btn-delete-position"
          onClick={handleDelete}
        >
          Delete
        </button>
      </div>

      {/* Close Position Modal */}
      {showCloseModal && (
        <div className="close-modal-overlay">
          <div className="close-modal">
            <h4>Close Position</h4>
            <p>Enter the exit price for {position.symbol}</p>
            <div className="close-modal-input">
              <label>Exit Price (SAR)</label>
              <input
                type="number"
                step="0.01"
                min="0"
                value={exitPrice}
                onChange={(e) => setExitPrice(e.target.value)}
                placeholder="Enter exit price"
                autoFocus
              />
            </div>
            {exitPrice && position.entry_price && (
              <div className="close-preview">
                <span>Expected P&L: </span>
                <span style={{
                  color: parseFloat(exitPrice) >= position.entry_price ? '#00c853' : '#ff1744'
                }}>
                  {((parseFloat(exitPrice) - position.entry_price) * position.quantity).toFixed(2)} SAR
                  ({(((parseFloat(exitPrice) - position.entry_price) / position.entry_price) * 100).toFixed(2)}%)
                </span>
              </div>
            )}
            <div className="close-modal-actions">
              <button
                className="btn-cancel"
                onClick={() => setShowCloseModal(false)}
                disabled={closing}
              >
                Cancel
              </button>
              <button
                className="btn-confirm"
                onClick={handleClose}
                disabled={closing || !exitPrice || exitPrice <= 0}
              >
                {closing ? 'Closing...' : 'Confirm Close'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PositionCard;
