"""Each metric against a small hand-worked example."""
import numpy as np
import pandas as pd
import pytest

from src import metrics as M

IDX = pd.date_range("2020-03-31", periods=6, freq="ME")
R = pd.Series([0.10, -0.05, 0.02, -0.10, 0.04, 0.03], index=IDX)
B = pd.Series([0.05, -0.02, 0.01, -0.04, 0.02, 0.01], index=IDX)
RF = pd.Series(0.001, index=IDX)


def test_growth_and_moments():
    end = 1.10 * 0.95 * 1.02 * 0.90 * 1.04 * 1.03          # 1.02158...
    s, _ = M.summary(R, RF, start=10_000)
    assert s["End balance"] == pytest.approx(10_000 * end)
    assert s["Annualised return (CAGR)"] == pytest.approx(end ** 2 - 1)   # 6 months -> 12/6 = 2
    assert s["Expected return"] == pytest.approx(12 * 0.04 / 6)
    assert s["Standard deviation"] == pytest.approx(np.sqrt(12) * R.std(ddof=1))


def test_drawdown_by_hand():
    # Peak 1.10 after month 1; trough 1.10*0.95*1.02*0.90 = 0.95931 after month 4.
    s, _ = M.summary(R, RF)
    assert s["Maximum drawdown"] == pytest.approx(0.95931 / 1.10 - 1, abs=1e-6)


def test_sharpe_and_sortino_by_hand():
    ex = R - 0.001                                           # [.099,-.051,.019,-.101,.039,.029]
    s, _ = M.summary(R, RF)
    assert s["Sharpe ratio (ex-post)"] == pytest.approx(np.sqrt(12) * ex.mean() / ex.std(ddof=1))
    dd = np.sqrt((0.051 ** 2 + 0.101 ** 2) / 6)              # divide by all 6 months, not the 2 losing ones
    assert s["Sortino ratio"] == pytest.approx(np.sqrt(12) * ex.mean() / dd)


def test_ex_ante_uses_the_model_inputs():
    s, _ = M.summary(R, RF, ex_ante=(0.15, 0.20))
    assert s["Sharpe ratio (ex-ante)"] == pytest.approx((0.15 - 0.012) / 0.20)
    s2, _ = M.summary(R, RF, ex_ante=(0.15, 0.20, 0.03))       # the rf at estimation, kept fixed
    assert s2["Sharpe ratio (ex-ante)"] == pytest.approx((0.15 - 0.03) / 0.20)


def test_relative_metrics_by_hand():
    a = R - B                                                # [.05,-.03,.01,-.06,.02,.02]
    s, _ = M.summary(R, RF, B)
    assert s["Active return"] == pytest.approx(12 * a.mean())
    assert s["Tracking error"] == pytest.approx(np.sqrt(12) * a.std(ddof=1))
    assert s["Information ratio"] == pytest.approx(s["Active return"] / s["Tracking error"])
    xb, xr = B - 0.001, R - 0.001
    assert s["Beta to SPY"] == pytest.approx(np.cov(xr, xb)[0, 1] / np.var(xb, ddof=1))
    assert s["Benchmark correlation"] == pytest.approx(np.corrcoef(R, B)[0, 1])


def test_cvar_takes_the_worst_five_percent_rounded_up():
    s, _ = M.summary(R, RF)                                  # ceil(0.05*6) = 1 month
    assert s["CVaR 95% (monthly)"] == pytest.approx(-0.10)
    r60 = pd.Series(np.linspace(-0.2, 0.2, 60), index=pd.date_range("2020-01-31", periods=60, freq="ME"))
    assert M.cvar(r60) == pytest.approx(np.sort(r60.values)[:3].mean())


def test_partial_years_are_labelled():
    r = pd.Series(0.01, index=pd.date_range("2020-03-31", "2021-12-31", freq="ME"))
    y = M.calendar_years(r)
    assert list(y.index) == ["2020 (Mar-Dec)", "2021"]
    assert y["2021"] == pytest.approx(1.01 ** 12 - 1)


def test_daily_drawdown_sees_what_month_ends_hide():
    days = pd.bdate_range("2020-02-28", "2020-04-30")
    v = pd.Series(100.0, index=days)
    v.loc["2020-03-10":"2020-03-25"] = 70.0                  # a crash that recovers before month end
    r = pd.Series(0.0, index=pd.to_datetime(["2020-03-31", "2020-04-30"]))
    s, _ = M.summary(r, pd.Series(0.0, index=r.index), daily_values=v)
    assert s["Maximum drawdown"] == 0.0 and s["Maximum drawdown (daily)"] == pytest.approx(-0.30)


def test_benchmark_column_and_formatting():
    t, lab = M.summary_table({"Portfolio": R, "SPY": B}, RF)
    assert t.loc["Beta to SPY", "SPY"] == 1.0 and np.isnan(t.loc["Active return", "SPY"])
    f = M.formatted(t, lab)
    assert f.loc["Start balance", "Portfolio"] == "$10,000"
    assert f.loc["Best year", "Portfolio"].endswith("(2020 (Mar-Aug))")
    assert f.loc["Active return", "SPY"] == "N/A"


def test_daily_window_starts_on_the_last_trading_day_before_the_first_month():
    days = pd.bdate_range("2020-02-25", "2020-03-31")          # 29 Feb 2020 is a Saturday
    v = pd.Series(100.0, index=days)
    v.loc["2020-03-02":] = 90.0                                # the fall happens over the weekend
    r = pd.Series([-0.10], index=pd.to_datetime(["2020-03-31"]))
    s, _ = M.summary(r, pd.Series(0.0, index=r.index), daily_values=v)
    assert s["Maximum drawdown (daily)"] == pytest.approx(-0.10)


def test_sharpe_half_width_matches_lo_2002_by_hand():
    import numpy as np, pandas as pd
    from src import metrics as M
    # 24 months alternating +3% / -1% excess: mean 1%, sd(ddof=1) = 2*sqrt(24/23)
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    r = pd.Series([0.03, -0.01] * 12, index=idx)
    rf = pd.Series(0.0, index=idx)
    sd = 0.02 * np.sqrt(24 / 23)
    s_m = 0.01 / sd
    expected = 1.959964 * np.sqrt(12) * np.sqrt((1 + s_m ** 2 / 2) / 24)
    assert np.isclose(M.sharpe_half_width(r, rf), expected)
    table, labels = M.summary_table({"P": r, "SPY": r * 0.5}, rf)
    shown = M.formatted(table, labels)
    lo, hi = (float(x) for x in shown.loc["Sharpe ratio, 95% half-width", "P"].split(" to "))
    assert np.isclose((lo + hi) / 2, table.loc["Sharpe ratio (ex-post)", "P"], atol=0.01)
