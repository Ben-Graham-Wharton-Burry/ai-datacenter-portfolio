import numpy as np
import pandas as pd
import pytest

from src import config as C, dataset as D
from scripts.refresh_data import parse_cpi, parse_ff5


@pytest.fixture(scope="module")
def ds():
    if not (D.CACHE / "prices_daily.csv").exists():
        pytest.skip("No data snapshot: run scripts/refresh_data.py first")
    return D.build()


def test_vertiv_shell_prices_are_removed(ds):
    assert ds.prices.loc[:"2020-02-06", "VRT"].isna().all()
    assert ds.prices.loc["2020-02-07":, "VRT"].notna().any()
    assert ds.monthly["VRT"].first_valid_index() == pd.Timestamp("2020-03-31")
    assert any(l.startswith("VRT") for l in ds.log)


def test_estimation_window_is_complete_and_starts_march_2020(ds):
    est = ds.est()
    assert est.index[0] == pd.Timestamp("2020-03-31") and est.index[-1] == pd.Timestamp("2024-12-31")
    assert len(est) == 58 and not est.isna().any().any()
    assert set(est.columns) == set(C.SINGLE_NAMES) and not set(C.THEMATIC_ETFS) & set(est.columns)


def test_out_of_sample_follows_without_overlap(ds):
    oos = ds.oos()
    assert oos.index[0] == pd.Timestamp("2025-01-31") and oos.index[0] > ds.est().index[-1]


def test_risk_free_is_monthly_and_time_varying(ds):
    rf = ds.rf.loc[ds.est().index]
    assert rf.notna().all() and rf.min() < 0.0005 and rf.max() > 0.004     # ~0% in 2021, ~5% in 2023
    ex = ds.excess(ds.est())
    assert np.allclose(ex["EQIX"], ds.est()["EQIX"] - rf)


def test_incomplete_final_month_is_dropped():
    idx = pd.bdate_range("2024-01-02", "2024-03-15")
    px = pd.DataFrame({"A": np.linspace(100, 110, len(idx))}, index=idx)
    m, log = D.monthly_returns(px)
    assert m.index[-1] == pd.Timestamp("2024-02-29") and "March 2024" in log[0]
    px2 = px.loc[:"2024-02-29"]
    m2, log2 = D.monthly_returns(px2)
    assert m2.index[-1] == pd.Timestamp("2024-02-29") and not log2


def test_missing_month_end_gives_missing_return_not_zero():
    idx = pd.bdate_range("2024-01-02", "2024-04-30")
    px = pd.DataFrame({"A": 100.0}, index=idx)
    px.loc["2024-02-01":"2024-02-29", "A"] = np.nan
    m, _ = D.monthly_returns(px)
    assert np.isnan(m.loc["2024-02-29", "A"]) and np.isnan(m.loc["2024-03-31", "A"])


def test_refresh_parsers():
    ff_text = ("This file was created ...\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n"
               "202401,   0.71,  -5.01,  -2.38,   0.68,  -0.86,   0.47\n"
               "202402,   5.06,  -0.78,  -3.49,  -1.89,  -2.31,   0.42\n\n"
               "  Annual Factors: January-December \n,Mkt-RF,SMB,HML,RMW,CMA,RF\n2024,  1, 2, 3, 4, 5, 6\n")
    ff = parse_ff5(ff_text)
    assert list(ff.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert ff.index[-1] == pd.Timestamp("2024-02-29") and ff.loc["2024-01-31", "RF"] == pytest.approx(0.0047)
    cpi = parse_cpi("observation_date,CPIAUCSL\n2024-01-01,309.7\n2024-02-01,311.0\n")
    assert cpi.index[0] == pd.Timestamp("2024-01-31") and cpi.iloc[-1] == 311.0


def test_market_cap_units_for_adrs_and_flags():
    from src import market_caps as MC
    # an ordinary US listing: shares x price
    r = MC.resolve("EQIX", 96e6, 943.0, 91.0)
    assert abs(r["market_cap_bn"] - 90.5) < 0.1 and r["flag"] == ""
    # GDS reported in ordinary shares: dividing by the ADS ratio gives the right size
    r = MC.resolve("GDS", 1.5e9, 23.7, 4.5)
    assert r["unit"].startswith("converted") and abs(r["market_cap_bn"] - 1.5e9 / 8 * 23.7 / 1e9) < 1e-9
    # GDS already reported in ADS: used as is
    assert MC.resolve("GDS", 190e6, 23.7, 4.5)["unit"] == "reported in ADS"
    # a large gap is kept but flagged for review
    assert MC.resolve("CAT", 480e6, 363.0, 140.0)["flag"].startswith("review")
