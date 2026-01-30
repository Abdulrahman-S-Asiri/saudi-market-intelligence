// Copyright (c) 2026 Abdulrahman Asiri.
// Engineered via Vibe Coding.
// Licensed under the MIT License.

/**
 * AddPositionModal Component
 * Form for manually adding new positions
 */

import React, { useState, useEffect } from 'react';

const AddPositionModal = ({ isOpen, onClose, onCreate, stocks }) => {
  const [formData, setFormData] = useState({
    symbol: '',
    entry_price: '',
    quantity: '1',
    notes: ''
  });
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isOpen) {
      // Reset form when modal opens
      setFormData({
        symbol: stocks.length > 0 ? stocks[0].symbol : '',
        entry_price: '',
        quantity: '1',
        notes: ''
      });
      setError(null);
    }
  }, [isOpen, stocks]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    // Validation
    if (!formData.symbol) {
      setError('Please select a stock');
      return;
    }
    if (!formData.entry_price || parseFloat(formData.entry_price) <= 0) {
      setError('Please enter a valid entry price');
      return;
    }
    if (!formData.quantity || parseFloat(formData.quantity) <= 0) {
      setError('Please enter a valid quantity');
      return;
    }

    setCreating(true);
    try {
      await onCreate({
        symbol: formData.symbol,
        entry_price: parseFloat(formData.entry_price),
        quantity: parseFloat(formData.quantity),
        notes: formData.notes || null,
        source: 'MANUAL'
      });
      onClose();
    } catch (err) {
      setError(err.message || 'Failed to create position');
    } finally {
      setCreating(false);
    }
  };

  if (!isOpen) return null;

  const selectedStock = stocks.find(s => s.symbol === formData.symbol);

  return (
    <div className="modal-overlay">
      <div className="add-position-modal">
        <div className="modal-header">
          <h3>Add New Position</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>

        <form onSubmit={handleSubmit}>
          {error && <div className="modal-error">{error}</div>}

          <div className="form-group">
            <label htmlFor="symbol">Stock Symbol</label>
            <select
              id="symbol"
              name="symbol"
              value={formData.symbol}
              onChange={handleChange}
              className="form-select"
            >
              <option value="">Select a stock...</option>
              {stocks.map(stock => (
                <option key={stock.symbol} value={stock.symbol}>
                  {stock.symbol} - {stock.name}
                </option>
              ))}
            </select>
            {selectedStock && (
              <span className="form-hint">{selectedStock.sector}</span>
            )}
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="entry_price">Entry Price (SAR)</label>
              <input
                type="number"
                id="entry_price"
                name="entry_price"
                value={formData.entry_price}
                onChange={handleChange}
                placeholder="0.00"
                step="0.01"
                min="0"
                className="form-input"
              />
            </div>

            <div className="form-group">
              <label htmlFor="quantity">Quantity</label>
              <input
                type="number"
                id="quantity"
                name="quantity"
                value={formData.quantity}
                onChange={handleChange}
                placeholder="1"
                step="1"
                min="1"
                className="form-input"
              />
            </div>
          </div>

          {formData.entry_price && formData.quantity && (
            <div className="form-total">
              <span>Total Value:</span>
              <span className="total-value">
                {(parseFloat(formData.entry_price) * parseFloat(formData.quantity)).toFixed(2)} SAR
              </span>
            </div>
          )}

          <div className="form-group">
            <label htmlFor="notes">Notes (Optional)</label>
            <textarea
              id="notes"
              name="notes"
              value={formData.notes}
              onChange={handleChange}
              placeholder="Add any notes about this position..."
              className="form-textarea"
              rows="2"
            />
          </div>

          <div className="modal-actions">
            <button
              type="button"
              className="btn-cancel"
              onClick={onClose}
              disabled={creating}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn-create"
              disabled={creating}
            >
              {creating ? 'Creating...' : 'Create Position'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddPositionModal;
