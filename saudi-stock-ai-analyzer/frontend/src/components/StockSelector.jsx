import React, { useState, useMemo } from 'react';
import Select, { components } from 'react-select';
import { motion, AnimatePresence } from 'framer-motion';

const sectorIcons = {
  'Energy': '/icons/energy.svg',
  'Banking': '/icons/banking.svg',
  'Chemicals': '/icons/chemicals.svg',
  'Telecommunications': '/icons/telecom.svg',
  'Mining': '/icons/mining.svg',
};

const sectorColors = {
  'Energy': '#ff9f43',
  'Banking': '#00d4ff',
  'Chemicals': '#a29bfe',
  'Telecommunications': '#00c853',
  'Mining': '#ff6b6b',
};

const CustomOption = ({ children, ...props }) => {
  const { data } = props;
  return (
    <components.Option {...props}>
      <div className="stock-option">
        <div className="stock-option-main">
          <span className="stock-symbol">{data.symbol}</span>
          <span className="stock-name">{data.name}</span>
        </div>
        <span
          className="stock-sector-badge"
          style={{ backgroundColor: `${sectorColors[data.sector]}20`, color: sectorColors[data.sector] }}
        >
          {data.sector}
        </span>
      </div>
    </components.Option>
  );
};

const CustomSingleValue = ({ children, ...props }) => {
  const { data } = props;
  return (
    <components.SingleValue {...props}>
      <div className="stock-selected">
        <span className="stock-symbol">{data.symbol}</span>
        <span className="stock-name">{data.name}</span>
      </div>
    </components.SingleValue>
  );
};

const GroupHeading = (props) => (
  <components.GroupHeading {...props}>
    <div className="group-heading">
      <span
        className="group-dot"
        style={{ backgroundColor: sectorColors[props.data.label] }}
      />
      {props.data.label}
    </div>
  </components.GroupHeading>
);

const StockSelector = ({ stocks, selectedStock, onSelect, onHover }) => {
  const [hoveredStock, setHoveredStock] = useState(null);

  // Group stocks by sector
  const groupedOptions = useMemo(() => {
    const groups = {};
    stocks.forEach(stock => {
      const sector = stock.sector || 'Other';
      if (!groups[sector]) {
        groups[sector] = {
          label: sector,
          options: [],
        };
      }
      groups[sector].options.push({
        value: stock.symbol,
        label: `${stock.symbol} - ${stock.name}`,
        ...stock,
      });
    });
    return Object.values(groups).sort((a, b) => a.label.localeCompare(b.label));
  }, [stocks]);

  const selectedOption = useMemo(() => {
    const stock = stocks.find(s => s.symbol === selectedStock);
    return stock ? { value: stock.symbol, label: `${stock.symbol} - ${stock.name}`, ...stock } : null;
  }, [stocks, selectedStock]);

  const customStyles = {
    control: (base, state) => ({
      ...base,
      background: 'rgba(26, 26, 46, 0.8)',
      borderColor: state.isFocused ? '#00d4ff' : 'rgba(255, 255, 255, 0.1)',
      borderRadius: '12px',
      padding: '4px 8px',
      boxShadow: state.isFocused ? '0 0 0 2px rgba(0, 212, 255, 0.2)' : 'none',
      '&:hover': {
        borderColor: '#00d4ff',
      },
      minWidth: '320px',
    }),
    menu: (base) => ({
      ...base,
      background: 'rgba(26, 26, 46, 0.95)',
      backdropFilter: 'blur(20px)',
      borderRadius: '12px',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      overflow: 'hidden',
      zIndex: 100,
    }),
    menuList: (base) => ({
      ...base,
      padding: '8px',
    }),
    option: (base, state) => ({
      ...base,
      background: state.isFocused
        ? 'rgba(0, 212, 255, 0.1)'
        : 'transparent',
      borderRadius: '8px',
      cursor: 'pointer',
      padding: '10px 12px',
      marginBottom: '4px',
      '&:active': {
        background: 'rgba(0, 212, 255, 0.2)',
      },
    }),
    singleValue: (base) => ({
      ...base,
      color: '#fff',
    }),
    input: (base) => ({
      ...base,
      color: '#fff',
    }),
    placeholder: (base) => ({
      ...base,
      color: '#888',
    }),
    group: (base) => ({
      ...base,
      paddingTop: '8px',
    }),
    groupHeading: (base) => ({
      ...base,
      color: '#888',
      fontSize: '11px',
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginBottom: '8px',
      padding: '0 8px',
    }),
  };

  return (
    <div className="stock-selector-container">
      <label className="selector-label">Select Stock</label>
      <Select
        value={selectedOption}
        onChange={(option) => onSelect(option.value)}
        options={groupedOptions}
        styles={customStyles}
        components={{
          Option: CustomOption,
          SingleValue: CustomSingleValue,
          GroupHeading,
        }}
        isSearchable
        placeholder="Search stocks..."
        noOptionsMessage={() => 'No stocks found'}
        onMenuOpen={() => setHoveredStock(null)}
      />

      <AnimatePresence>
        {hoveredStock && (
          <motion.div
            className="stock-preview glass-card"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
          >
            <div className="preview-header">
              <span className="preview-symbol">{hoveredStock.symbol}</span>
              <span className="preview-name">{hoveredStock.name}</span>
            </div>
            <div className="preview-stats">
              <span className="preview-sector">{hoveredStock.sector}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default StockSelector;
