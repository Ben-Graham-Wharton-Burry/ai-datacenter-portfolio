"""Configuration for the AI Data Center Portfolio project."""
import numpy as np

CLUSTERS = {
    'REIT':           ['EQIX', 'DLR', 'AMT', 'CCI', 'IRM'],
    'Asia':           ['GDS', 'VNET'],
    'Infrastructure': ['VRT', 'ANET', 'CSCO', 'MRVL', 'SBGSY', 'ETN', 'JCI', 'HON', 'GNRC', 'CAT'],
    'ETF':            ['DTCR', 'SRVR'],
}
ALL_TICKERS = [t for tickers in CLUSTERS.values() for t in tickers]
BENCHMARK = 'SPY'
TICKER_CLUSTER = {}
for cluster, tickers in CLUSTERS.items():
    for t in tickers:
        TICKER_CLUSTER[t] = cluster

START_DATE = '2019-01-01'
END_DATE   = '2026-03-01'
TRAIN_END    = '2022-12-31'
EMBARGO_MONTHS = 1
TEST_START   = '2023-02-01'

RF_ANNUAL  = 0.045
RF_MONTHLY = (1 + RF_ANNUAL) ** (1/12) - 1

LOOKBACK_MONTHS = 24
TRANSACTION_COST_BPS = 10
SLIPPAGE_BPS = 5
MAX_WEIGHT = 0.15

WF_MIN_TRAIN_MONTHS = 24
WF_EXPANDING = True

N_BOOTSTRAP = 10_000
BLOCK_SIZE  = 6
SIGNIFICANCE_LEVEL = 0.05

COLORS = {
    'navy': '#1B2A4A', 'blue': '#3B7DD8', 'gold': '#C8A951',
    'teal': '#2CA58D', 'red': '#D64045', 'gray': '#888888',
    'green': '#4CAF50', 'purple': '#7B68EE', 'orange': '#FF8C00',
}
PALETTE = list(COLORS.values())
