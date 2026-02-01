import axios from 'axios';

const apiClient = axios.create({
    baseURL: 'http://localhost:8000/api', // Vite proxy will handle /api if strictly used, but here we might want absolute or relative
    headers: {
        'Content-Type': 'application/json',
    },
});

// If using Vite proxy (which is set to /api -> http://localhost:8000)
// We can just use /api as base if we want, or keep localhost:8000 for explicit cors.
// Let's stick to the existing pattern but encapsulated.

export const fetchStocks = async () => {
    const response = await apiClient.get('/stocks');
    return response.data;
};

export const fetchStockAnalysis = async (symbol, period = '6mo', trainModel = true) => {
    const response = await apiClient.get(`/analyze/${symbol}`, {
        params: { period, train_model: trainModel }
    });
    return response.data;
};

export const fetchChartData = async (symbol, period = '6mo') => {
    const response = await apiClient.get(`/chart/${symbol}`, {
        params: { period }
    });
    return response.data;
};
