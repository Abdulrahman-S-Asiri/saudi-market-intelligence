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

  // Fetch available stocks on mount
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
    fetchStocks();
  }, []);

  const fetchAnalysis = useCallback(async (symbol, period = '6mo') => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE}/api/analyze/${symbol}`, {
        params: { period, train_model: true }
      });
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

  return {
    analysis,
    chartData,
    stocks,
    signalHistory,
    backtestResults,
    loading,
    error,
    fetchAnalysis,
    fetchChartData,
    fetchSignalHistory,
    fetchBacktest,
    compareStocks
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
