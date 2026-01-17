import React from 'react';

const MetricsCard = ({ performance, periodReturns, trend }) => {
  if (!performance) return null;

  const formatValue = (value, suffix = '') => {
    if (value === null || value === undefined) return 'N/A';
    return `${value}${suffix}`;
  };

  const getValueColor = (value) => {
    if (value > 0) return '#00c853';
    if (value < 0) return '#ff1744';
    return '#ffffff';
  };

  return (
    <div className="metrics-card">
      <h3>Performance Metrics</h3>

      <div className="metrics-grid">
        <div className="metric-item">
          <span className="metric-label">Total Return</span>
          <span
            className="metric-value"
            style={{ color: getValueColor(performance.total_return) }}
          >
            {formatValue(performance.total_return, '%')}
          </span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Volatility</span>
          <span className="metric-value">{formatValue(performance.volatility, '%')}</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Sharpe Ratio</span>
          <span className="metric-value">{formatValue(performance.sharpe_ratio)}</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Max Drawdown</span>
          <span className="metric-value" style={{ color: '#ff1744' }}>
            -{formatValue(performance.max_drawdown, '%')}
          </span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Win Rate</span>
          <span className="metric-value">{formatValue(performance.win_rate, '%')}</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Trading Days</span>
          <span className="metric-value">{performance.trading_days}</span>
        </div>
      </div>

      {trend && (
        <div className="trend-section">
          <h4>Trend Analysis</h4>
          <div className="trend-info">
            <span className={`trend-badge ${trend.direction.toLowerCase().replace('_', '-')}`}>
              {trend.direction.replace('_', ' ')}
            </span>
            <span className="trend-strength">Strength: {trend.strength}%</span>
          </div>
          <div className="period-changes">
            <span>
              5D: <span style={{ color: getValueColor(trend.change_5d) }}>
                {trend.change_5d > 0 ? '+' : ''}{trend.change_5d}%
              </span>
            </span>
            <span>
              20D: <span style={{ color: getValueColor(trend.change_20d) }}>
                {trend.change_20d > 0 ? '+' : ''}{trend.change_20d}%
              </span>
            </span>
          </div>
        </div>
      )}

      {periodReturns && (
        <div className="period-returns">
          <h4>Period Returns</h4>
          <div className="returns-grid">
            {['1d', '5d', '1m', '3m', '6m'].map(period => (
              <div key={period} className="return-item">
                <span className="period-label">{period.toUpperCase()}</span>
                <span
                  className="period-value"
                  style={{ color: getValueColor(periodReturns[`return_${period}`]) }}
                >
                  {periodReturns[`return_${period}`] !== null
                    ? `${periodReturns[`return_${period}`] > 0 ? '+' : ''}${periodReturns[`return_${period}`]}%`
                    : 'N/A'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default MetricsCard;
