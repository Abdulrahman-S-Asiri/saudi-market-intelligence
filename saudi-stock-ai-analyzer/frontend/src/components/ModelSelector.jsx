import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const modelIcons = {
  'lstm': (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
    </svg>
  ),
  'ensemble': (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3" />
      <circle cx="4" cy="8" r="2" />
      <circle cx="20" cy="8" r="2" />
      <circle cx="4" cy="16" r="2" />
      <circle cx="20" cy="16" r="2" />
      <path d="M6 8l3.5 2.5M15 10.5L18 8M6 16l3.5-2.5M15 13.5l3 2.5" />
    </svg>
  ),
  'chronos': (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
      <path d="M8 3h8M6 21h12" />
    </svg>
  ),
};

const modelColors = {
  'lstm': '#00d4ff',
  'ensemble': '#a29bfe',
  'chronos': '#ff9f43',
};

const ModelSelector = ({
  models = [],
  currentModel,
  onSelectModel,
  loading = false,
  disabled = false
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectingModel, setSelectingModel] = useState(null);

  const handleModelSelect = async (modelType) => {
    if (disabled || loading || modelType === currentModel) return;

    const model = models.find(m => m.type === modelType);
    if (!model?.available) return;

    setSelectingModel(modelType);
    try {
      await onSelectModel(modelType);
    } finally {
      setSelectingModel(null);
      setIsExpanded(false);
    }
  };

  const currentModelData = models.find(m => m.type === currentModel) || {
    type: currentModel,
    name: currentModel?.toUpperCase() || 'Unknown',
    available: true
  };

  return (
    <div className="model-selector-container">
      <label className="selector-label">Prediction Model</label>

      <div className="model-selector-wrapper">
        <button
          className={`model-selector-button ${isExpanded ? 'expanded' : ''}`}
          onClick={() => setIsExpanded(!isExpanded)}
          disabled={disabled || loading}
          style={{
            borderColor: isExpanded ? modelColors[currentModel] : 'rgba(255, 255, 255, 0.1)'
          }}
        >
          <span className="model-icon" style={{ color: modelColors[currentModel] }}>
            {modelIcons[currentModel]}
          </span>
          <span className="model-name">{currentModelData.name}</span>
          <span className={`model-chevron ${isExpanded ? 'rotated' : ''}`}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </span>
        </button>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              className="model-dropdown"
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.15 }}
            >
              {models.map((model) => (
                <button
                  key={model.type}
                  className={`model-option ${model.type === currentModel ? 'selected' : ''} ${!model.available ? 'disabled' : ''}`}
                  onClick={() => handleModelSelect(model.type)}
                  disabled={!model.available || selectingModel === model.type}
                >
                  <span className="model-icon" style={{ color: model.available ? modelColors[model.type] : '#666' }}>
                    {modelIcons[model.type]}
                  </span>
                  <div className="model-info">
                    <span className="model-name">{model.name}</span>
                    <span className="model-description">{model.description}</span>
                    {!model.available && model.note && (
                      <span className="model-note">{model.note}</span>
                    )}
                  </div>
                  {model.type === currentModel && (
                    <span className="model-check">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    </span>
                  )}
                  {selectingModel === model.type && (
                    <span className="model-loading">
                      <svg className="spinner" width="16" height="16" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="60" strokeLinecap="round" />
                      </svg>
                    </span>
                  )}
                  {!model.requires_training && model.available && (
                    <span className="model-badge">Zero-shot</span>
                  )}
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <style jsx>{`
        .model-selector-container {
          position: relative;
          min-width: 280px;
        }

        .selector-label {
          display: block;
          color: #888;
          font-size: 12px;
          font-weight: 500;
          margin-bottom: 8px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .model-selector-wrapper {
          position: relative;
        }

        .model-selector-button {
          display: flex;
          align-items: center;
          gap: 12px;
          width: 100%;
          padding: 12px 16px;
          background: rgba(26, 26, 46, 0.8);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          color: #fff;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .model-selector-button:hover:not(:disabled) {
          border-color: rgba(255, 255, 255, 0.2);
          background: rgba(26, 26, 46, 0.9);
        }

        .model-selector-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .model-selector-button.expanded {
          border-radius: 12px 12px 0 0;
        }

        .model-icon {
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .model-name {
          flex: 1;
          text-align: left;
          font-weight: 500;
        }

        .model-chevron {
          display: flex;
          align-items: center;
          color: #888;
          transition: transform 0.2s ease;
        }

        .model-chevron.rotated {
          transform: rotate(180deg);
        }

        .model-dropdown {
          position: absolute;
          top: 100%;
          left: 0;
          right: 0;
          background: rgba(26, 26, 46, 0.95);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-top: none;
          border-radius: 0 0 12px 12px;
          overflow: hidden;
          z-index: 100;
        }

        .model-option {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          width: 100%;
          padding: 14px 16px;
          background: transparent;
          border: none;
          color: #fff;
          text-align: left;
          cursor: pointer;
          transition: background 0.15s ease;
        }

        .model-option:hover:not(.disabled):not(.selected) {
          background: rgba(0, 212, 255, 0.1);
        }

        .model-option.selected {
          background: rgba(0, 212, 255, 0.15);
        }

        .model-option.disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .model-info {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .model-info .model-name {
          font-weight: 500;
          font-size: 14px;
        }

        .model-description {
          font-size: 12px;
          color: #888;
        }

        .model-note {
          font-size: 11px;
          color: #ff6b6b;
          margin-top: 2px;
        }

        .model-check {
          color: #00d4ff;
          display: flex;
          align-items: center;
        }

        .model-loading .spinner {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .model-badge {
          font-size: 10px;
          padding: 2px 6px;
          background: rgba(255, 159, 67, 0.2);
          color: #ff9f43;
          border-radius: 4px;
          font-weight: 600;
          text-transform: uppercase;
        }
      `}</style>
    </div>
  );
};

export default ModelSelector;
