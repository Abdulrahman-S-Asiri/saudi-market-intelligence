"""
Data Loader Module for Saudi Stock Market
Fetches stock data from Saudi Arabian stock market (Tadawul)
Includes macroeconomic data: Brent Oil and TASI Index
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import os

# Cache for macro data to avoid repeated downloads
_macro_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
MACRO_CACHE_DURATION = timedelta(hours=1)


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

    def fetch_brent_oil(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Fetch Brent Crude Oil prices (BZ=F)

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified

        Returns:
            DataFrame with Brent Oil data (Date, Close)
        """
        global _macro_cache
        cache_key = f"brent_oil_{start_date}_{end_date}_{period}"

        # Check cache
        if cache_key in _macro_cache:
            cached_data, cached_time = _macro_cache[cache_key]
            if datetime.now() - cached_time < MACRO_CACHE_DURATION:
                return cached_data.copy()

        try:
            symbol = "BZ=F"  # Brent Crude Oil Futures
            if start_date and end_date:
                oil_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            else:
                oil_data = yf.download(symbol, period=period, progress=False)

            if oil_data.empty:
                print(f"[WARNING] No Brent Oil data found, using empty DataFrame")
                return pd.DataFrame(columns=['Date', 'Oil_Close'])

            oil_data.reset_index(inplace=True)

            # Handle multi-level columns
            if isinstance(oil_data.columns, pd.MultiIndex):
                oil_data.columns = [col[0] if isinstance(col, tuple) else col for col in oil_data.columns]

            # Rename columns for clarity
            oil_data = oil_data[['Date', 'Close']].copy()
            oil_data.columns = ['Date', 'Oil_Close']

            # Cache the result
            _macro_cache[cache_key] = (oil_data.copy(), datetime.now())

            return oil_data

        except Exception as e:
            print(f"[WARNING] Error fetching Brent Oil data: {str(e)}")
            return pd.DataFrame(columns=['Date', 'Oil_Close'])

    def fetch_tasi_index(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Fetch TASI Index data (^TASI.SR)

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified

        Returns:
            DataFrame with TASI data (Date, Close)
        """
        global _macro_cache
        cache_key = f"tasi_index_{start_date}_{end_date}_{period}"

        # Check cache
        if cache_key in _macro_cache:
            cached_data, cached_time = _macro_cache[cache_key]
            if datetime.now() - cached_time < MACRO_CACHE_DURATION:
                return cached_data.copy()

        try:
            symbol = "^TASI.SR"  # TASI Index
            if start_date and end_date:
                tasi_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            else:
                tasi_data = yf.download(symbol, period=period, progress=False)

            if tasi_data.empty:
                print(f"[WARNING] No TASI data found, using empty DataFrame")
                return pd.DataFrame(columns=['Date', 'TASI_Close'])

            tasi_data.reset_index(inplace=True)

            # Handle multi-level columns
            if isinstance(tasi_data.columns, pd.MultiIndex):
                tasi_data.columns = [col[0] if isinstance(col, tuple) else col for col in tasi_data.columns]

            # Rename columns for clarity
            tasi_data = tasi_data[['Date', 'Close']].copy()
            tasi_data.columns = ['Date', 'TASI_Close']

            # Cache the result
            _macro_cache[cache_key] = (tasi_data.copy(), datetime.now())

            return tasi_data

        except Exception as e:
            print(f"[WARNING] Error fetching TASI data: {str(e)}")
            return pd.DataFrame(columns=['Date', 'TASI_Close'])

    def fetch_macro_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Fetch all macroeconomic data (Brent Oil + TASI Index)

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified

        Returns:
            DataFrame with Date, Oil_Close, TASI_Close columns
        """
        # Fetch both datasets
        oil_data = self.fetch_brent_oil(start_date, end_date, period)
        tasi_data = self.fetch_tasi_index(start_date, end_date, period)

        if oil_data.empty and tasi_data.empty:
            return pd.DataFrame(columns=['Date', 'Oil_Close', 'TASI_Close'])

        # Merge on Date
        if not oil_data.empty and not tasi_data.empty:
            macro_data = pd.merge(oil_data, tasi_data, on='Date', how='outer')
        elif not oil_data.empty:
            macro_data = oil_data.copy()
            macro_data['TASI_Close'] = None
        else:
            macro_data = tasi_data.copy()
            macro_data['Oil_Close'] = None

        # Sort by date
        macro_data = macro_data.sort_values('Date').reset_index(drop=True)

        # Forward fill to handle weekend gaps (Saudi vs Global markets)
        macro_data['Oil_Close'] = macro_data['Oil_Close'].ffill()
        macro_data['TASI_Close'] = macro_data['TASI_Close'].ffill()

        return macro_data

    def fetch_stock_with_macro(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Fetch stock data merged with macroeconomic data

        Args:
            symbol: Stock symbol (e.g., '2222' for Aramco)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if dates not specified

        Returns:
            DataFrame with stock data + Oil_Close + TASI_Close columns
        """
        # Fetch stock data
        stock_data = self.fetch_stock_data(symbol, start_date, end_date, period)

        if stock_data.empty:
            return stock_data

        # Handle multi-level columns from yfinance
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]

        # Fetch macro data
        macro_data = self.fetch_macro_data(start_date, end_date, period)

        if macro_data.empty:
            # Return stock data without macro features
            stock_data['Oil_Close'] = None
            stock_data['TASI_Close'] = None
            return stock_data

        # Ensure Date columns are datetime
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        macro_data['Date'] = pd.to_datetime(macro_data['Date'])

        # Merge stock data with macro data
        merged_data = pd.merge(
            stock_data,
            macro_data,
            on='Date',
            how='left'
        )

        # Forward fill macro data to handle weekend gaps
        merged_data['Oil_Close'] = merged_data['Oil_Close'].ffill().bfill()
        merged_data['TASI_Close'] = merged_data['TASI_Close'].ffill().bfill()

        return merged_data


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


def verify_macro_data():
    """Test macro data fetching (Brent Oil + TASI Index)"""
    print("\n" + "=" * 60)
    print("Macroeconomic Data Loader - Verification Test")
    print("=" * 60)

    loader = SaudiStockDataLoader()

    # Test Brent Oil
    print("\n1. Fetching Brent Oil (BZ=F)...")
    oil_data = loader.fetch_brent_oil(period="1mo")
    if not oil_data.empty:
        print(f"   [OK] Brent Oil: {len(oil_data)} rows")
        print(f"   Latest Oil Price: ${oil_data['Oil_Close'].iloc[-1]:.2f}")
    else:
        print("   [WARNING] No Brent Oil data")

    # Test TASI Index
    print("\n2. Fetching TASI Index (^TASI.SR)...")
    tasi_data = loader.fetch_tasi_index(period="1mo")
    if not tasi_data.empty:
        print(f"   [OK] TASI Index: {len(tasi_data)} rows")
        print(f"   Latest TASI: {tasi_data['TASI_Close'].iloc[-1]:.2f}")
    else:
        print("   [WARNING] No TASI data")

    # Test combined macro data
    print("\n3. Fetching combined macro data...")
    macro_data = loader.fetch_macro_data(period="1mo")
    if not macro_data.empty:
        print(f"   [OK] Combined: {len(macro_data)} rows")
        print(f"   Columns: {list(macro_data.columns)}")
    else:
        print("   [WARNING] No macro data")

    # Test stock with macro data
    print("\n4. Fetching Saudi Aramco with macro data...")
    stock_with_macro = loader.fetch_stock_with_macro("2222", period="3mo")
    if not stock_with_macro.empty:
        print(f"   [OK] Stock + Macro: {len(stock_with_macro)} rows")
        print(f"   Columns: {list(stock_with_macro.columns)}")
        print(f"\n   Sample data:")
        print(stock_with_macro[['Date', 'Close', 'Oil_Close', 'TASI_Close']].tail(5).to_string())
    else:
        print("   [ERROR] Failed to fetch stock with macro data")

    return True


if __name__ == "__main__":
    verify_data_loader()
    verify_macro_data()
