import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { TrashIcon, XMarkIcon } from '@heroicons/react/24/outline';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000/api',
});

const fetchPositions = async (status) => {
  const params = status ? { status } : {};
  const response = await apiClient.get('/positions', { params });
  return response.data;
};

const fetchSummary = async () => {
  const response = await apiClient.get('/positions/summary');
  return response.data;
};

const closePosition = async ({ positionId, exitPrice }) => {
  const response = await apiClient.put(`/positions/${positionId}/close`, {
    exit_price: exitPrice,
    exit_reason: 'Manual close'
  });
  return response.data;
};

const deletePosition = async (positionId) => {
  const response = await apiClient.delete(`/positions/${positionId}`);
  return response.data;
};

const Portfolio = () => {
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState('');
  const [closeModal, setCloseModal] = useState(null);
  const [exitPrice, setExitPrice] = useState('');

  const { data: positions, isLoading } = useQuery({
    queryKey: ['positions', statusFilter],
    queryFn: () => fetchPositions(statusFilter || undefined),
  });

  const { data: summary } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: fetchSummary,
  });

  const closeMutation = useMutation({
    mutationFn: closePosition,
    onSuccess: () => {
      queryClient.invalidateQueries(['positions']);
      queryClient.invalidateQueries(['portfolio-summary']);
      setCloseModal(null);
      setExitPrice('');
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deletePosition,
    onSuccess: () => {
      queryClient.invalidateQueries(['positions']);
      queryClient.invalidateQueries(['portfolio-summary']);
    },
  });

  const handleClosePosition = () => {
    if (!closeModal || !exitPrice) return;
    closeMutation.mutate({ positionId: closeModal.id, exitPrice: parseFloat(exitPrice) });
  };

  const handleDeletePosition = (positionId) => {
    if (window.confirm('Are you sure you want to delete this position?')) {
      deleteMutation.mutate(positionId);
    }
  };

  return (
    <main className="px-6 max-w-[1920px] mx-auto animate-enter">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-2">Portfolio</h1>
        <p className="text-gray-400">Manage your positions and track performance</p>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="glass-card p-4">
            <div className="text-xs text-gray-400 mb-1">Open Positions</div>
            <div className="text-2xl font-bold text-white">{summary.open_positions || 0}</div>
          </div>
          <div className="glass-card p-4">
            <div className="text-xs text-gray-400 mb-1">Total Invested</div>
            <div className="text-2xl font-bold text-white font-mono">
              {summary.total_invested?.toLocaleString() || 0} SAR
            </div>
          </div>
          <div className="glass-card p-4">
            <div className="text-xs text-gray-400 mb-1">Unrealized P/L</div>
            <div className={`text-2xl font-bold font-mono ${
              (summary.unrealized_pnl || 0) >= 0 ? 'text-green-500' : 'text-red-500'
            }`}>
              {(summary.unrealized_pnl || 0) >= 0 ? '+' : ''}{summary.unrealized_pnl?.toFixed(2) || 0} SAR
            </div>
          </div>
          <div className="glass-card p-4">
            <div className="text-xs text-gray-400 mb-1">Win Rate</div>
            <div className="text-2xl font-bold text-white">
              {summary.win_rate?.toFixed(1) || 0}%
            </div>
          </div>
        </div>
      )}

      {/* Filter */}
      <div className="glass-card p-4 mb-6 flex items-center gap-4">
        <label className="text-sm text-gray-400">Filter:</label>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="bg-dark-800 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-brand-primary"
        >
          <option value="">All Positions</option>
          <option value="OPEN">Open Only</option>
          <option value="CLOSED">Closed Only</option>
        </select>
      </div>

      {/* Positions Table */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-8 h-8 border-2 border-brand-primary border-t-transparent rounded-full animate-spin"></div>
        </div>
      ) : (
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-white/5">
                <tr>
                  <th className="text-left text-xs text-gray-400 font-medium px-4 py-3">Symbol</th>
                  <th className="text-left text-xs text-gray-400 font-medium px-4 py-3">Type</th>
                  <th className="text-right text-xs text-gray-400 font-medium px-4 py-3">Entry</th>
                  <th className="text-right text-xs text-gray-400 font-medium px-4 py-3">Qty</th>
                  <th className="text-right text-xs text-gray-400 font-medium px-4 py-3">Stop Loss</th>
                  <th className="text-right text-xs text-gray-400 font-medium px-4 py-3">Take Profit</th>
                  <th className="text-center text-xs text-gray-400 font-medium px-4 py-3">Status</th>
                  <th className="text-right text-xs text-gray-400 font-medium px-4 py-3">P/L</th>
                  <th className="text-center text-xs text-gray-400 font-medium px-4 py-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {positions && positions.length > 0 ? (
                  positions.map((position) => (
                    <tr key={position.id} className="border-t border-white/5 hover:bg-white/5">
                      <td className="px-4 py-3">
                        <span className="font-bold text-white">{position.symbol}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`text-xs px-2 py-1 rounded ${
                          position.position_type === 'LONG' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'
                        }`}>
                          {position.position_type}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-white">
                        {position.entry_price?.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-white">
                        {position.quantity?.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-red-400">
                        {position.stop_loss?.toFixed(2) || '--'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-green-400">
                        {position.take_profit?.toFixed(2) || '--'}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className={`text-xs px-2 py-1 rounded ${
                          position.status === 'OPEN' ? 'bg-blue-500/20 text-blue-500' : 'bg-gray-500/20 text-gray-400'
                        }`}>
                          {position.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        {position.realized_pnl !== undefined && position.realized_pnl !== null ? (
                          <span className={`font-mono ${position.realized_pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                            {position.realized_pnl >= 0 ? '+' : ''}{position.realized_pnl?.toFixed(2)}
                          </span>
                        ) : (
                          <span className="text-gray-500">--</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <div className="flex items-center justify-center gap-2">
                          {position.status === 'OPEN' && (
                            <button
                              onClick={() => {
                                setCloseModal(position);
                                setExitPrice(position.entry_price?.toString() || '');
                              }}
                              className="text-xs px-2 py-1 bg-yellow-500/20 text-yellow-500 rounded hover:bg-yellow-500/30"
                            >
                              Close
                            </button>
                          )}
                          <button
                            onClick={() => handleDeletePosition(position.id)}
                            className="p-1 text-red-400 hover:text-red-300"
                          >
                            <TrashIcon className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="9" className="px-4 py-8 text-center text-gray-500">
                      No positions found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Close Position Modal */}
      {closeModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="glass-card p-6 w-full max-w-md">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-white">Close Position</h3>
              <button onClick={() => setCloseModal(null)} className="text-gray-400 hover:text-white">
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <p className="text-gray-400 mb-4">
              Closing {closeModal.position_type} position for {closeModal.symbol}
            </p>
            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-2">Exit Price</label>
              <input
                type="number"
                step="0.01"
                value={exitPrice}
                onChange={(e) => setExitPrice(e.target.value)}
                className="w-full bg-dark-800 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-brand-primary"
                placeholder="Enter exit price"
              />
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setCloseModal(null)}
                className="flex-1 px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20"
              >
                Cancel
              </button>
              <button
                onClick={handleClosePosition}
                disabled={!exitPrice || closeMutation.isPending}
                className="flex-1 px-4 py-2 bg-brand-primary text-white rounded-lg hover:bg-brand-primary/80 disabled:opacity-50"
              >
                {closeMutation.isPending ? 'Closing...' : 'Confirm Close'}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
};

export default Portfolio;
