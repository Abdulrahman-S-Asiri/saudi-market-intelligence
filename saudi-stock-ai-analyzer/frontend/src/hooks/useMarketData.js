import { useQuery } from '@tanstack/react-query';
import { fetchStocks, fetchStockAnalysis, fetchChartData, fetchMarketRankings } from '../api/client';

export const useStocks = () => {
    return useQuery({
        queryKey: ['stocks'],
        queryFn: fetchStocks,
        staleTime: 1000 * 60 * 60, // 1 hour
    });
};

export const useStockAnalysis = (symbol, period = '6mo', enabled = true) => {
    return useQuery({
        queryKey: ['analysis', symbol, period],
        queryFn: () => fetchStockAnalysis(symbol, period, true),
        enabled: !!symbol && enabled,
        staleTime: 1000 * 60 * 5, // 5 minutes
        retry: 1,
    });
};

export const useChartData = (symbol, period = '6mo', enabled = true) => {
    return useQuery({
        queryKey: ['chart', symbol, period],
        queryFn: () => fetchChartData(symbol, period),
        enabled: !!symbol && enabled,
        staleTime: 1000 * 60 * 5, // 5 minutes
    });
};

export const useMarketRankings = (quickScan = true) => {
    return useQuery({
        queryKey: ['market-rankings', quickScan],
        queryFn: () => fetchMarketRankings(quickScan),
        staleTime: 1000 * 60 * 5, // 5 minutes
        refetchInterval: 1000 * 60 * 15, // Refetch every 15 mins
    });
};
