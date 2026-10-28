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

# --------------------------------------------------------------------------
# Rebuild (October 2026): universe roles, windows and data corrections
# --------------------------------------------------------------------------
# The portfolio is built from single names only. The two thematic ETFs are
# passive benchmarks: the question is whether selecting and weighting names
# beats simply buying the theme. SPY is the market benchmark.
SINGLE_NAMES = [t for c in ('REIT', 'Asia', 'Infrastructure') for t in CLUSTERS[c]]
THEMATIC_ETFS = list(CLUSTERS['ETF'])
BENCHMARKS = [BENCHMARK] + THEMATIC_ETFS

# Estimation: March 2020 is the first month with a return for every single
# name (Vertiv lists on 7 Feb 2020, so its first full-month return is March).
# The window keeps the COVID crash month. Out of sample starts after the
# thesis was written.
EST_START, EST_END = '2020-03-31', '2024-12-31'
OOS_START = '2025-01-31'

# Prices that do not belong to the company the ticker now represents.
# Vertiv went public by merging into a SPAC, GS Acquisition Holdings; before
# the merger closed the ticker traded as that cash shell (flat near $9.70).
CORPORATE_EVENTS = {
    'VRT': ('2020-02-07', 'Listed via SPAC merger with GS Acquisition Holdings; '
                          'earlier prices are the cash shell, not Vertiv'),
}

START_DATE = '2019-01-01'
END_DATE   = '2026-03-01'
TRAIN_END    = '2022-12-31'
EMBARGO_MONTHS = 1
TEST_START   = '2023-02-01'

# Legacy constant used by the original notebooks. The rebuild uses the monthly
# one-month T-bill rate from the Fama-French file instead (2019-2024 average
# 2.3% a year, near zero in 2020-21): a fixed 4.5% understated every Sharpe ratio.
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


# Company, what it does for data centres, and where its primary listing is. Used for the universe table.
DESCRIPTIONS = {
    "EQIX": ("Equinix", "Colocation and interconnection", "US"),
    "DLR": ("Digital Realty Trust", "Wholesale and colocation", "US"),
    "AMT": ("American Tower", "Cell towers; edge data centres (CoreSite)", "US"),
    "CCI": ("Crown Castle", "Cell towers and fibre", "US"),
    "IRM": ("Iron Mountain", "Records storage; data centres", "US"),
    "GDS": ("GDS Holdings", "Data centres in China", "China, US ADS"),
    "VNET": ("VNET Group", "Data centres in China", "China, US ADS"),
    "VRT": ("Vertiv", "Power and cooling equipment", "US"),
    "ANET": ("Arista Networks", "Data-centre network switching", "US"),
    "CSCO": ("Cisco Systems", "Networking equipment", "US"),
    "MRVL": ("Marvell Technology", "Data-centre chips: custom silicon, optics", "US"),
    "SBGSY": ("Schneider Electric", "Electrical distribution and cooling", "France, US ADR"),
    "ETN": ("Eaton", "Power management and distribution", "US"),
    "JCI": ("Johnson Controls", "Building systems and cooling", "US"),
    "HON": ("Honeywell", "Building automation and controls", "US"),
    "GNRC": ("Generac", "Backup generators", "US"),
    "CAT": ("Caterpillar", "Generator sets and power systems", "US"),
    "SPY": ("SPDR S&P 500 ETF", "Benchmark: the US equity market", "US"),
    "DTCR": ("Global X Data Center ETF", "Comparison: data-centre thematic ETF", "US"),
    "SRVR": ("Pacer Data & Infra. ETF", "Comparison: data-centre REIT ETF", "US"),
}
