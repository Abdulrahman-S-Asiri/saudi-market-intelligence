"""
Data Loader Module for Saudi Stock Market
Fetches stock data from Saudi Arabian stock market (Tadawul)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List


class SaudiStockDataLoader:
    """
    Data loader for Saudi stock market (Tadawul)
    Saudi stocks on Yahoo Finance use .SR suffix (e.g., 2222.SR for Aramco)
    """

    def __init__(self):
        """Initialize the data loader"""
        self.market_suffix = ".SR"

    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch stock data for a Saudi stock

        Args:
            symbol: Stock symbol (e.g., '2222' for Aramco, will auto-add .SR)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            DataFrame with stock data (Date, Open, High, Low, Close, Volume, Adj Close)
        """
        # Add .SR suffix if not present
        if not symbol.endswith(self.market_suffix):
            symbol = f"{symbol}{self.market_suffix}"

        try:
            # Fetch data from yfinance
            if start_date and end_date:
                stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
            else:
                stock = yf.download(symbol, period=period, progress=False)

            if stock.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Reset index to make Date a column
            stock.reset_index(inplace=True)

            return stock

        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")

    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y"
    ) -> dict:
        """
        Fetch data for multiple Saudi stocks

        Args:
            symbols: List of stock symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified

        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}

        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, start_date, end_date, period)
                stock_data[symbol] = data
                print(f"[OK] Successfully fetched data for {symbol}")
            except Exception as e:
                print(f"[FAIL] Failed to fetch {symbol}: {str(e)}")
                stock_data[symbol] = None

        return stock_data

    def get_stock_info(self, symbol: str) -> dict:
        """
        Get detailed information about a stock

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock information
        """
        if not symbol.endswith(self.market_suffix):
            symbol = f"{symbol}{self.market_suffix}"

        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            raise Exception(f"Error fetching info for {symbol}: {str(e)}")


def verify_data_loader():
    """Quick verification test for the data loader"""
    print("=" * 60)
    print("Saudi Stock Data Loader - Verification Test")
    print("=" * 60)

    loader = SaudiStockDataLoader()

    # Test with Saudi Aramco (2222.SR)
    test_symbol = "2222"  # Saudi Aramco
    print(f"\nFetching data for {test_symbol} (Saudi Aramco)...")

    try:
        # Fetch last 30 days of data
        data = loader.fetch_stock_data(test_symbol, period="1mo")

        print(f"\n[SUCCESS] Data fetched successfully!")
        print(f"  Rows: {len(data)}")
        print(f"  Columns: {list(data.columns)}")
        print(f"\n  First 5 rows:")
        print(data.head())

        # Get the closing price (handle multi-level columns)
        close_col = [col for col in data.columns if 'Close' in str(col)]
        if close_col:
            latest_price = data[close_col[0]].iloc[-1]
            print(f"\n  Latest closing price: {float(latest_price):.2f} SAR")

        return True

    except Exception as e:
        print(f"\n[ERROR] Verification failed: {str(e)}")
        return False


if __name__ == "__main__":
    verify_data_loader()
