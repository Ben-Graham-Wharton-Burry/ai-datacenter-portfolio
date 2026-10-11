"""Data acquisition with caching and return computation."""
import os, pandas as pd, numpy as np, yfinance as yf
from typing import List, Tuple

class DataLoader:
    def __init__(self, cache_dir='./data_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load_prices(self, tickers: List[str], start: str, end: str, use_cache=True) -> pd.DataFrame:
        """
        Load adjusted close prices, preferring a frozen snapshot over a live download.

        Resolution order:
          1. data_cache/prices_daily.csv   the committed snapshot, written by fetch_data.py
          2. data_cache/prices.csv         a previous run's incremental cache
          3. yfinance                      live download

        The snapshot takes priority so that the notebook reproduces the same numbers
        on every run and on every machine. A notebook that re-downloads on execution
        will drift away from the figures committed alongside it, and nothing in the
        output will say so.
        """
        snapshot = os.path.join(self.cache_dir, 'prices_daily.csv')
        if use_cache and os.path.exists(snapshot):
            prices = pd.read_csv(snapshot, index_col=0, parse_dates=True)
            available = [t for t in tickers if t in prices.columns]
            missing = [t for t in tickers if t not in prices.columns]
            print(f"Loaded frozen snapshot: {len(available)}/{len(tickers)} tickers, "
                  f"{len(prices)} trading days "
                  f"({prices.index[0].date()} to {prices.index[-1].date()})")
            if missing:
                print(f"  not in snapshot: {missing}")
            return prices[available].dropna(how='all')

        cache_path = os.path.join(self.cache_dir, 'prices.csv')
        if use_cache and os.path.exists(cache_path):
            prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            missing = [t for t in tickers if t not in prices.columns]
            if not missing and prices.index[-1] >= pd.Timestamp(end) - pd.Timedelta(days=7):
                return prices[[t for t in tickers if t in prices.columns]].dropna(how='all')
        print(f"Downloading {len(tickers)} tickers from yfinance...")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw.to_frame() if isinstance(raw, pd.Series) else raw
        prices = prices.dropna(axis=1, how='all').ffill(limit=5).dropna(how='all')
        prices.to_csv(cache_path)
        available = [t for t in tickers if t in prices.columns]
        print(f"Loaded {len(available)}/{len(tickers)} tickers, {len(prices)} trading days")
        return prices[available]

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        daily = prices.pct_change().dropna(how='all')
        monthly_prices = prices.resample('ME').last()
        monthly = monthly_prices.pct_change().dropna(how='all')
        return daily, monthly

    @staticmethod
    def split_by_date(returns: pd.DataFrame, train_end: str, test_start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = returns.loc[:train_end].copy()
        test  = returns.loc[test_start:].copy()
        print(f"Train: {train.index[0].strftime('%Y-%m')} to {train.index[-1].strftime('%Y-%m')} ({len(train)} obs)")
        print(f"Test:  {test.index[0].strftime('%Y-%m')} to {test.index[-1].strftime('%Y-%m')} ({len(test)} obs)")
        return train, test
