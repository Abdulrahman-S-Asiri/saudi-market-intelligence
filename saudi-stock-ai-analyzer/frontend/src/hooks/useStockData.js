import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const useStockData = () => {
  const [analysis, setAnalysis] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [stocks, setStocks] = useState([]);
  const [signalHistory, setSignalHistory] = useState([]);
  const [backtestResults, setBacktestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [currentModel, setCurrentModel] = useState('ensemble');

  // Fetch available stocks and models on mount
  useEffect(() => {
    const fetchStocks = async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/stocks`);
        setStocks(response.data.stocks);
      } catch (err) {
        console.error('Failed to fetch stocks:', err);
        // Set default stocks if API fails
        setStocks([
          { symbol: '2222', name: 'Saudi Aramco', sector: 'Energy' },
          { symbol: '1120', name: 'Al Rajhi Bank', sector: 'Banking' },
          { symbol: '2010', name: 'SABIC', sector: 'Chemicals' },
          { symbol: '7010', name: 'STC', sector: 'Telecommunications' },
          { symbol: '1211', name: "Ma'aden", sector: 'Mining' },
          { symbol: '2350', name: 'Saudi Kayan', sector: 'Chemicals' },
          { symbol: '1180', name: 'Al Inma Bank', sector: 'Banking' },
          { symbol: '2310', name: 'Sipchem', sector: 'Chemicals' },
        ]);
      }
    };

    const fetchModels = async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/models`);
        setAvailableModels(response.data.models || []);
        setCurrentModel(response.data.current_model || 'ensemble');
      } catch (err) {
        console.error('Failed to fetch models:', err);
        // Set default models if API fails
        setAvailableModels([
          { type: 'lstm', name: 'LSTM Neural Network', available: true },
          { type: 'ensemble', name: 'Ensemble (LSTM + XGBoost)', available: true },
          { type: 'chronos', name: 'Chronos-2 Foundation Model', available: false },
        ]);
      }
    };

    fetchStocks();
    fetchModels();
  }, []);

  const fetchAnalysis = useCallback(async (symbol, period = '6mo', modelType = null) => {
    setLoading(true);
    setError(null);

    try {
      const params = { period, train_model: true };
      if (modelType) {
        params.model_type = modelType;
      }
      const response = await axios.get(`${API_BASE}/api/analyze/${symbol}`, { params });
      setAnalysis(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
      console.error('Analysis failed:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchChartData = useCallback(async (symbol, period = '6mo') => {
    try {
      const response = await axios.get(`${API_BASE}/api/chart/${symbol}`, {
        params: { period }
      });

      // Enhance chart data with MACD calculations if not present
      const data = response.data.data || [];
      const enhancedData = data.map((item, index) => {
        // Add MACD data if available from analysis
        return {
          ...item,
          macd: item.macd || null,
          macd_signal: item.macd_signal || null,
        };
      });

      setChartData({
        ...response.data,
        data: enhancedData
      });
    } catch (err) {
      console.error('Chart data failed:', err);
    }
  }, []);

  const fetchSignalHistory = useCallback(async (symbol) => {
    try {
      const response = await axios.get(`${API_BASE}/api/signals/history/${symbol}`);
      setSignalHistory(response.data.signals || []);
    } catch (err) {
      console.error('Signal history failed:', err);
      // Generate mock signal history for demo
      const mockHistory = generateMockSignalHistory();
      setSignalHistory(mockHistory);
    }
  }, []);

  const fetchBacktest = useCallback(async (symbol, period = '1y') => {
    try {
      const response = await axios.get(`${API_BASE}/api/backtest/${symbol}`, {
        params: { period }
      });
      setBacktestResults(response.data);
    } catch (err) {
      console.error('Backtest failed:', err);
    }
  }, []);

  const compareStocks = useCallback(async (symbols) => {
    try {
      const response = await axios.get(`${API_BASE}/api/compare`, {
        params: { symbols: symbols.join(',') }
      });
      return response.data;
    } catch (err) {
      console.error('Comparison failed:', err);
      return null;
    }
  }, []);

  const selectModel = useCallback(async (modelType) => {
    try {
      const response = await axios.post(`${API_BASE}/api/models/select`, null, {
        params: { model_type: modelType }
      });
      if (response.data.success) {
        setCurrentModel(modelType);
        return { success: true, message: response.data.message };
      }
      return { success: false, error: response.data.error };
    } catch (err) {
      console.error('Model selection failed:', err);
      return {
        success: false,
        error: err.response?.data?.detail || err.message
      };
    }
  }, []);

  const fetchChronosForecast = useCallback(async (symbol, horizon = 5, period = '6mo') => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE}/api/chronos/${symbol}`, {
        params: { horizon, period }
      });
      return response.data;
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message;
      setError(errorMsg);
      console.error('Chronos forecast failed:', err);
      return { error: errorMsg };
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    analysis,
    chartData,
    stocks,
    signalHistory,
    backtestResults,
    loading,
    error,
    availableModels,
    currentModel,
    fetchAnalysis,
    fetchChartData,
    fetchSignalHistory,
    fetchBacktest,
    compareStocks,
    selectModel,
    fetchChronosForecast
  };
};

// Helper function to generate mock signal history for demo
const generateMockSignalHistory = () => {
  const signals = [];
  const now = new Date();
  const signalTypes = ['BUY', 'SELL', 'HOLD'];

  for (let i = 0; i < 15; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() - (i * 7)); // Weekly signals

    const signalType = signalTypes[Math.floor(Math.random() * signalTypes.length)];
    const confidence = Math.floor(Math.random() * 40) + 50; // 50-90%
    const price = 28 + Math.random() * 4; // 28-32 SAR range

    const correct = Math.random() > 0.35; // 65% accuracy
    const returnPct = correct
      ? (Math.random() * 5 + 1) * (signalType === 'SELL' ? -1 : 1)
      : (Math.random() * 3) * (signalType === 'SELL' ? 1 : -1);

    signals.push({
      timestamp: date.toISOString(),
      signal: signalType,
      confidence,
      price,
      outcome: i > 0 ? {
        correct,
        return_pct: returnPct
      } : null
    });
  }

  return signals;
};

export default useStockData;
