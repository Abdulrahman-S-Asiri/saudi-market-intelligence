import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock the custom hooks
vi.mock('./hooks/useMarketData', () => ({
    useStocks: vi.fn(() => ({ data: { stocks: [] } })),
    useStockAnalysis: vi.fn(() => ({ data: null, isLoading: false, error: null })),
    useChartData: vi.fn(() => ({ data: null }))
}));

vi.mock('./hooks/usePositions', () => ({
    default: vi.fn(() => ({
        fetchPositions: vi.fn(),
        createFromSignal: vi.fn()
    }))
}));

const queryClient = new QueryClient({
    defaultOptions: {
        queries: { retry: false },
    },
});

describe('App', () => {
    it('renders without crashing', () => {
        // We mock the child components that might cause issues or make complex requests
        render(
            <QueryClientProvider client={queryClient}>
                <App />
            </QueryClientProvider>
        );
        // Use a text matcher for something we know is in the header
        expect(screen.getByText(/TASI AI/i)).toBeInTheDocument();
    });
});
