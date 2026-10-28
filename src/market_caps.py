"""
Market capitalisations on 31 Dec 2024 for the Black-Litterman prior.

Market cap = shares outstanding x unadjusted closing price, both as close to
31 Dec 2024 as the data allow. refresh_data.py downloads the inputs from
Yahoo Finance and writes data_cache/market_caps_2024.csv; the estimators read
that file and fall back to the hand-entered reference figures only if it is
missing.

ADRs
----
Three names trade in the US as American depositary shares (ADS), where one
ADS represents a fixed number of ordinary shares. Yahoo is not consistent
about whether it reports ordinary shares or ADS for these tickers, so both
readings are computed. The data decide between them only as a check of
units: the two readings differ by the ADS ratio (8x, 6x or 5x), so the
right one is the one within a factor of two of the reference figure. A name
where neither reading qualifies is flagged for manual review, never guessed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data_cache" / "market_caps_2024.csv"
AS_OF = "2024-12-31"

# Ordinary shares represented by one ADS, from each depositary agreement.
ADR_RATIO = {
    "GDS": 8.0,       # GDS Holdings: 1 ADS = 8 Class A ordinary shares
    "VNET": 6.0,      # VNET Group: 1 ADS = 6 Class A ordinary shares
    "SBGSY": 0.2,     # Schneider Electric: 1 ADR = 0.2 ordinary share (5 ADRs per share)
}
FLAG_DEVIATION = 0.10    # report names more than 10% away from the reference figure


def resolve(ticker: str, shares: float, price: float, reference_bn: float) -> dict:
    """Market cap in USD billions, with the unit choice for ADRs made explicit."""
    as_reported = shares * price / 1e9
    ratio = ADR_RATIO.get(ticker)
    if ratio is None:
        cap, unit = as_reported, "ordinary shares"
    else:
        as_ads = shares / ratio * price / 1e9
        within = lambda v: 0.5 <= v / reference_bn <= 2.0
        if within(as_ads) and not within(as_reported):
            cap, unit = as_ads, f"converted to ADS (1 ADS = {ratio:g} shares)"
        elif within(as_reported) and not within(as_ads):
            cap, unit = as_reported, "reported in ADS"
        else:
            return {"market_cap_bn": float("nan"), "unit": "ambiguous", "deviation": float("nan"),
                    "flag": "review: neither ADS reading is near the reference"}
    dev = cap / reference_bn - 1
    flag = f"review: {dev:+.0%} from reference" if abs(dev) > FLAG_DEVIATION else ""
    return {"market_cap_bn": cap, "unit": unit, "deviation": dev, "flag": flag}


def load(path: Path = PATH) -> dict | None:
    """Market caps from the refreshed file; a name with no resolved value falls back to its reference figure."""
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="ticker")
    caps = df["market_cap_bn"].where(df["market_cap_bn"].notna(), df["reference_bn"])
    return {t: float(v) for t, v in caps.items()}
