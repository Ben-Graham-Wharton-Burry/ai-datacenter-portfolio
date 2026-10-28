"""Build every exhibit for the document from the frozen data snapshot.

Run from the project folder:  python scripts/build_exhibits.py

Each exhibit is written to exhibits/ as a vector PDF (for the document), a
PNG (for previews) and, for tables, a CSV with the unformatted numbers.
Exhibit numbers follow the order of the report: 1 to 14 in the main body, A and B in the appendices.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                      # noqa: E402
import pandas as pd                                                      # noqa: E402

from src import config as C, dataset as D, engine as EN, estimators as ES  # noqa: E402
from src import correlation as CR, montecarlo as MCS, exhibits as X, house_style as H, metrics as M, optimiser as OP  # noqa: E402

OUT = ROOT / "exhibits"
START_VALUE = 500_000
EST_FROM, EST_TO = "2020-02-29", C.EST_END          # the month-end before the first return, and the last
TABLE_ROWS = ["Start balance", "End balance", "Annualised return (CAGR)", "Expected return", "Standard deviation",
              "Best year", "Worst year", "Maximum drawdown", "Maximum drawdown (daily)", "Sharpe ratio (ex-ante)",
              "Sharpe ratio (ex-post)", "Sortino ratio", "Beta to SPY", "CVaR 95% (monthly)", "Active return",
              "Tracking error", "Information ratio", "Benchmark correlation"]
COLOURS = {"Black-Litterman": H.GREEN, "Historical mean": H.STEEL, "SPY": H.SILVER}
# Clusters kept as cautionary cases: the exhibit says so in its title and takeaway, rather than hiding the result.
CAUTIONARY = {"Asia": "a concentration case"}


def fit(est: pd.DataFrame, rf: float, cap: float = 1.0) -> dict:
    """Both models on one set of assets: weights and the optimiser's own expectations."""
    S_lw, _ = ES.ledoit_wolf_cc(est)
    bl = ES.black_litterman(S_lw, T=len(est))
    hist = OP.max_sharpe(ES.historical_mean(est), ES.sample_cov(est), rf, cap)
    blo = OP.max_sharpe(bl.posterior, bl.cov_post, 0.0, cap)
    return {"Historical mean": (hist.weights, hist.exp_return, hist.exp_vol),
            "Black-Litterman": (blo.weights, blo.exp_return + rf, blo.exp_vol)}


FULL_CAP = 0.15
MIX_COLOURS = {"Market-cap prior": H.SILVER, "Black-Litterman": H.GREEN, "Historical mean": H.STEEL}


def backtest(ds, models: dict, rf: float):
    """In-sample backtests of each model plus SPY: monthly returns, daily values, ex-ante tuples."""
    series, daily, ex = {}, {}, {}
    for model, (w, mu, sig) in models.items():
        bt = EN.run(ds.prices, w, EST_FROM, EST_TO)
        series[model], daily[model], ex[model] = bt.monthly, bt.daily, (mu, sig, rf)
    series["SPY"] = ds.monthly["SPY"].loc[series["Black-Litterman"].index]
    daily["SPY"] = ds.prices["SPY"].loc[daily["Black-Litterman"].index[0]:EST_TO].dropna()
    return series, daily, ex


def performance_and_wealth(ds, models: dict, rf: float, *, name: str, section: str, stem: str,
                           n_table: str, n_wealth: str, wealth_title: str, wealth_lead: str = "") -> dict:
    """The performance table and the growth-of-$500,000 chart for one set of models. Returns end values."""
    series, daily, ex = backtest(ds, models, rf)
    table, labels = M.summary_table(series, ds.rf, start=START_VALUE, daily_values=daily, ex_ante=ex)
    table = table[["Black-Litterman", "Historical mean", "SPY"]]
    table.loc[TABLE_ROWS].to_csv(OUT / f"{n_table}_{stem}_performance.csv")
    shown = M.formatted(table, labels).loc[TABLE_ROWS]
    fig = X.performance_table(shown, number=n_table, focal="Black-Litterman", section=section,
                              title=f"{name} against the S&P 500, Mar 2020 to Dec 2024",
                              takeaway="In-sample: weights estimated on the same period they are scored on. "
                                       "Annual rebalancing, 15 bps round-trip costs.")
    H.save(fig, OUT / f"{n_table}_{stem}_performance")

    end = {k: START_VALUE * v.iloc[-1] / v.iloc[0] for k, v in daily.items()}
    takeaway = wealth_lead or (f"Black-Litterman ${end['Black-Litterman'] / 1e3:,.0f}k, historical mean "
                               f"${end['Historical mean'] / 1e3:,.0f}k, S&P 500 ${end['SPY'] / 1e3:,.0f}k "
                               "at end-2024. In-sample, daily values.")
    fig = X.wealth_chart(daily, start_value=START_VALUE, number=n_wealth, colours=COLOURS, section=section,
                         title=wealth_title, takeaway=takeaway)
    H.save(fig, OUT / f"{n_wealth}_{stem}_wealth")
    return {"end": end, "daily": daily}


def cluster_exhibits(ds, name: str, first_number: int, prefix: str = "") -> None:
    tickers = C.CLUSTERS[name]
    est = ds.est(tickers)
    rf = float(ds.rf.loc[est.index].mean() * 12)
    models = fit(est, rf)
    n1, n2, n3 = (f"{prefix}{first_number + k}" for k in range(3))
    stem, section = name.lower(), f"{name} cluster"

    wb, wh = models["Black-Litterman"][0], models["Historical mean"][0]
    fig = X.weights_comparison(wb, wh, number=n1, title=f"{name} cluster: maximum Sharpe weights",
                               focal_label="Black-Litterman", ref_label="Historical mean", section=section)
    H.save(fig, OUT / f"{n1}_{stem}_weights")

    lead = ""
    title = f"{name} cluster: growth of $500,000"
    if name in CAUTIONARY:
        _, daily, _ = backtest(ds, models, rf)
        end = {k: START_VALUE * v.iloc[-1] / v.iloc[0] for k, v in daily.items()}
        dd = M.max_drawdown(daily["Black-Litterman"])
        title += f", {CAUTIONARY[name]}"
        lead = (f"{len(tickers)} names, fully invested: neither model can diversify. Peak-to-trough fall of "
                f"{abs(dd):.0%}, ${end['Black-Litterman'] / 1e3:,.0f}k left vs ${end['SPY'] / 1e3:,.0f}k in the S&P 500.")
    performance_and_wealth(ds, models, rf, name=f"{name} cluster", section=section, stem=stem,
                           n_table=n2, n_wealth=n3, wealth_title=title, wealth_lead=lead)
    print(f"{name}: exhibits {n1} to {n3}")


def full_portfolio_exhibits(ds, first_number: int) -> None:
    """Exhibits for the 17-name portfolio: weights, the cap, cluster mix, frontier, table, wealth."""
    est = ds.est()                                       # all single names
    rf = float(ds.rf.loc[est.index].mean() * 12)
    capped, uncapped = fit(est, rf, FULL_CAP), fit(est, rf, 1.0)
    n = [str(first_number + k) for k in range(6)]
    section, stem = "Full portfolio", "full"
    cap_txt = f"{FULL_CAP:.0%} cap"

    # 1. Weights, both models under the cap
    wb, wh = capped["Black-Litterman"][0], capped["Historical mean"][0]
    fig = X.weights_comparison(wb, wh, number=n[0], cap=FULL_CAP, section=section,
                               title=f"Full portfolio: maximum Sharpe weights with a {cap_txt}",
                               focal_label="Black-Litterman", ref_label="Historical mean")
    H.save(fig, OUT / f"{n[0]}_{stem}_weights")

    # 2. What the cap does to the focal model
    wbu = uncapped["Black-Litterman"][0]
    whu = uncapped["Historical mean"][0]
    fig = X.weights_comparison(wb, wbu, number=n[1], cap=FULL_CAP, section=section, ref_colour=H.SILVER,
                               title="Black-Litterman weights before and after the cap",
                               focal_label=f"With {cap_txt}", ref_label="Uncapped",
                               note=f"The same cap takes the historical mean from {OP.effective_n(whu):.1f} to {OP.effective_n(wh):.1f}.")
    H.save(fig, OUT / f"{n[1]}_{stem}_cap")

    # 3. Cluster mix: what the optimiser chose, set against the market-cap prior
    w_mkt = ES.market_weights(est.columns)
    mix = pd.DataFrame({"Market-cap prior": w_mkt, "Black-Litterman": wb, "Historical mean": wh})
    mix = mix.groupby(pd.Series(C.TICKER_CLUSTER)).sum().loc[["Infrastructure", "REIT", "Asia"]]
    fig = X.cluster_mix(mix, number=n[2], section=section, colours=MIX_COLOURS,
                        title="Cluster mix chosen by the optimiser, not imposed",
                        takeaway=f"No cluster budgets. Black-Litterman holds {(mix.loc['REIT', 'Black-Litterman'] - mix.loc['REIT', 'Market-cap prior']) * 100:.0f} points "
                                 f"more in REITs than the prior and holds {mix.loc['Asia', 'Black-Litterman']:.0%} in Asia. "
                                 f"Both models with a {cap_txt}.")
    H.save(fig, OUT / f"{n[2]}_{stem}_cluster_mix")
    mix.to_csv(OUT / f"{n[2]}_{stem}_cluster_mix.csv")

    # 4. Ex-ante frontiers, each in its own model's expected returns (total, not excess)
    S_lw, _ = ES.ledoit_wolf_cc(est)
    bl = ES.black_litterman(S_lw, T=len(est))
    mu_h, S_h = ES.historical_mean(est), ES.sample_cov(est)
    fr = {"Historical mean": OP.frontier(mu_h, S_h, FULL_CAP),
          "Black-Litterman": OP.frontier(bl.posterior + rf, bl.cov_post, FULL_CAP)}
    opt = {k: (capped[k][2], capped[k][1]) for k in ("Historical mean", "Black-Litterman")}
    assets = pd.DataFrame({"vol": np.sqrt(np.diag(S_h)), "ret": mu_h}, index=est.columns)
    spy = ds.monthly["SPY"].loc[est.index]
    fig = X.frontier_chart(fr, opt, assets, colours=COLOURS, number=n[3], section=section,
                           benchmark=("SPY", float(spy.std() * np.sqrt(12)), float(spy.mean() * 12)),
                           label_at={"EQIX": (-4, 3, "right"), "CSCO": (-4, -5, "right"), "DLR": (4, 3, "left"),
                                     "Black-Litterman": (-12, -22, "right")},
                           title=f"Two views of the same 17 stocks: efficient frontiers with a {cap_txt}",
                           takeaway=f"Ex-ante. Historical mean expects {opt['Historical mean'][1]:.0%} a year, "
                                    f"Black-Litterman {opt['Black-Litterman'][1]:.0%}. Hollow dots: each stock's "
                                    "historical mean and volatility.")
    H.save(fig, OUT / f"{n[3]}_{stem}_frontier")

    # 5 and 6. Performance table and growth of $500,000
    out = performance_and_wealth(ds, capped, rf, name=f"Full portfolio ({cap_txt})", section=section,
                                 stem=stem, n_table=n[4], n_wealth=n[5],
                                 wealth_title="Full portfolio: growth of $500,000")
    print(f"Full portfolio: exhibits {n[0]} to {n[5]}", {k: round(v) for k, v in out["end"].items()})


def appendix_exhibits(ds) -> None:
    """B1 clustering tree, B2 sample vs shrunk correlations, B3 out-of-sample correlations and change."""
    est, oos = ds.est(), ds.oos()
    R = est.corr()
    Z = CR.tree(R)
    order = CR.order(R)
    agr = CR.agreement(R, C.TICKER_CLUSTER)
    agr.table.assign(random_95th=agr.random_95).to_csv(OUT / "B1_clustering_sensitivity.csv")
    ari = agr.table.loc[CR.METHOD, "ARI"]
    fig = X.dendrogram_chart(Z, list(R.index), C.TICKER_CLUSTER, CR.cut_height(Z, 3), number="B1",
                             title="Do the returns sort into our three clusters?",
                             takeaway=f"Average linkage. Asia separates under every method; REITs and infrastructure "
                                      f"do not (ARI {ari:.2f}, random labels reach {agr.random_95:.2f}).")
    H.save(fig, OUT / "B1_clustering")

    Rs, delta = CR.shrunk_corr(est)
    rbar = CR.mean_offdiag(R)
    pd.concat({"sample": R.loc[order, order], "shrunk": Rs.loc[order, order]}).to_csv(OUT / "B2_correlations.csv")
    fig = X.correlation_panels([("Sample correlation", R, 1.0, "Correlation"),
                                ("Ledoit-Wolf shrunk (used by the optimiser)", Rs, 1.0, "Correlation")],
                               order, number="B2", title="What shrinkage does to the correlation matrix",
                               takeaway=f"Every pair moves {delta:.0%} of the way to the average correlation of "
                                        f"{rbar:.2f}. Monthly returns, Mar 2020 to Dec 2024, clustered order.")
    H.save(fig, OUT / "B2_correlations")

    Ro = oos.corr()
    span = f"{oos.index[0]:%b %Y} to {oos.index[-1]:%b %Y}"
    dR = Ro - R
    lim = float(np.ceil(np.nanmax(np.abs(dR.to_numpy())) * 5) / 5)        # round up to a step of 0.2
    pd.concat({"oos": Ro.loc[order, order], "change": dR.loc[order, order]}).to_csv(OUT / "B3_oos_correlations.csv")
    fig = X.correlation_panels([(f"Correlation, {span}", Ro, 1.0, "Correlation"),
                                ("Change from the estimation window", dR, lim, "Change in correlation")],
                               order, number="B3", title="Correlations after the estimation window",
                               takeaway=f"Provisional: {len(oos)} months for {oos.shape[1]} stocks is a noisy estimate. "
                                        f"Average correlation {rbar:.2f} in sample, {CR.mean_offdiag(Ro):.2f} after.",
                               source=f"Source: Yahoo Finance total-return prices, {span}; author's calculations.")
    H.save(fig, OUT / "B3_oos_correlations")
    print("Appendix B: exhibits B1 to B3;", f"ARI {ari:.2f} vs random {agr.random_95:.2f}; delta {delta:.2f}")


OOS_FROM, OOS_TO = C.EST_END, "2026-09-30"            # formation date; last complete month in the snapshot
OOS_COLOURS = {"Black-Litterman": H.GREEN, "Historical mean": H.STEEL, "Equal weight (1/N)": H.AMBER,
               "SPY": H.SILVER, "DTCR": H.SILVER, "SRVR": H.SILVER}
OOS_DASHES = {"DTCR": (0, (5, 3)), "SRVR": (0, (1.5, 2.5))}
OOS_HEADERS = {"Black-Litterman": "Black-\nLitterman", "Historical mean": "Historical\nmean",
               "Equal weight (1/N)": "Equal\nweight (1/N)", "SPY": "SPY", "DTCR": "DTCR", "SRVR": "SRVR"}


def oos_exhibits(ds, first_number: int) -> dict:
    """Weights frozen at end-2024, rebalanced annually to the same targets, scored from Jan 2025."""
    est = ds.est()
    rf0 = float(ds.rf.loc[est.index].mean() * 12)            # the rf the models were formed with
    models = fit(est, rf0, FULL_CAP)
    names = list(est.columns)
    weights = {"Black-Litterman": models["Black-Litterman"][0], "Historical mean": models["Historical mean"][0],
               "Equal weight (1/N)": pd.Series(1 / len(names), index=names)}
    n = [str(first_number + k) for k in range(4)]
    section = "Out of sample"

    series, daily, ex = {}, {}, {}
    for k, w in weights.items():
        bt = EN.run(ds.prices, w, OOS_FROM, OOS_TO)
        series[k], daily[k] = bt.monthly, bt.daily
        if k in models:
            ex[k] = (models[k][1], models[k][2], rf0)
    idx = series["Black-Litterman"].index
    first_day = daily["Black-Litterman"].index[0]
    for t in C.BENCHMARKS:                                   # SPY, DTCR, SRVR: buy and hold, no costs
        series[t] = ds.monthly[t].loc[idx]
        daily[t] = ds.prices[t].loc[first_day:OOS_TO].dropna()
    span = f"{idx[0]:%b %Y} to {idx[-1]:%b %Y}"
    source = f"Source: Yahoo Finance total-return prices, {span}; author's calculations."

    # 1. performance table, calendar-year rows in place of best and worst year
    table, labels = M.summary_table(series, ds.rf, start=START_VALUE, daily_values=daily, ex_ante=ex)
    order = list(OOS_COLOURS)
    table = table[order]
    cal = pd.DataFrame({k: M.calendar_years(series[k]) for k in order})
    rows = [r for r in TABLE_ROWS if r not in ("Best year", "Worst year")]
    at = rows.index("Standard deviation") + 1
    rows = rows[:at] + list(cal.index) + rows[at:]
    rows.insert(rows.index("Sharpe ratio (ex-post)") + 1, "Sharpe ratio, 95% half-width")
    full = pd.concat([table, cal])
    full.loc[rows].to_csv(OUT / f"{n[0]}_oos_performance.csv")
    shown = M.formatted(table, labels)
    for y in cal.index:
        shown.loc[y] = [f"{v:.2%}" for v in cal.loc[y]]
    shown = shown.loc[rows].rename(index=M.DISPLAY_NAMES).rename(columns=OOS_HEADERS)
    shown.index = [f"Return {i}" if i in cal.index else i for i in shown.index]
    b, sp = table.loc["Sharpe ratio (ex-post)", "Black-Litterman"], table.loc["Sharpe ratio (ex-post)", "SPY"]
    hw = table.loc["Sharpe ratio, 95% half-width", "Black-Litterman"]
    fig = X.performance_table(shown, number=n[0], focal=OOS_HEADERS["Black-Litterman"], section=section,
                              source=source, first_col=0.28,
                              title=f"Out of sample, {span}: weights fixed at end-2024",
                              takeaway=f"{len(idx)} months. Black-Litterman's Sharpe ratio of {b:.2f} has a 95% interval "
                                       f"of ±{hw:.2f}; SPY's is {sp:.2f}. Annual rebalancing, 15 bps round trip.")
    H.save(fig, OUT / f"{n[0]}_oos_performance")

    # 2. growth of $500,000
    end = {k: START_VALUE * v.iloc[-1] / v.iloc[0] for k, v in daily.items()}
    best = max(end, key=end.get)
    fig = X.wealth_chart(daily, start_value=START_VALUE, number=n[1], colours=OOS_COLOURS, dashes=OOS_DASHES,
                         section=section, source=source, title=f"Growth of $500,000 after the estimation window",
                         takeaway=f"Invested 31 Dec 2024 at end-2024 weights. Highest end value: {best}, "
                                  f"${end[best] / 1e3:,.0f}k. Dashed: thematic ETFs.")
    H.save(fig, OUT / f"{n[1]}_oos_wealth")

    # 3. underwater
    uw = {k: daily[k] for k in ("Black-Litterman", "Historical mean", "Equal weight (1/N)", "SPY")}
    worst = min(uw, key=lambda k: M.max_drawdown(uw[k]))
    fig = X.drawdown_chart(uw, number=n[2], colours=OOS_COLOURS, section=section, source=source,
                           title="How far each portfolio fell from its peak",
                           takeaway=f"Daily values. Deepest fall: {worst}, {M.max_drawdown(uw[worst]):.0%}. "
                                    "Dots mark each trough.")
    H.save(fig, OUT / f"{n[2]}_oos_drawdown")

    # 4. forecasts against outcomes, with the sampling error of the outcomes
    T = len(idx)
    rows = {}
    for k in ("Black-Litterman", "Historical mean", "Equal weight (1/N)", "SPY"):
        vol, ret = float(series[k].std() * np.sqrt(12)), float(series[k].mean() * 12)
        rows[k] = {"ex_ante_return": models[k][1] if k in models else np.nan,
                   "realised_return": ret, "ret_hw": 1.96 * vol / np.sqrt(T / 12),        # SE of a mean: sigma / sqrt(years)
                   "ex_ante_vol": models[k][2] if k in models else np.nan,
                   "realised_vol": vol, "vol_hw": 1.96 * vol / np.sqrt(2 * (T - 1))}  # SE of a sd: sigma / sqrt(2(n-1))
    fc = pd.DataFrame(rows).T
    fc.to_csv(OUT / f"{n[3]}_oos_forecast_check.csv")
    h = fc.loc["Historical mean"]
    fig = X.forecast_check(fc, colours=OOS_COLOURS, number=n[3], section=section, source=source,
                           title="Which forecasts can 21 months test?",
                           takeaway=f"Return intervals are too wide to judge either model. Volatility is not: the historical "
                                    f"mean expected {h.ex_ante_vol:.0%} and realised {h.realised_vol:.0%}.")
    H.save(fig, OUT / f"{n[3]}_oos_forecast_check")
    pts = fc
    print(f"Out of sample: exhibits {n[0]} to {n[3]}", {k: round(v) for k, v in end.items()})
    return {"table": table, "end": end, "points": pts}


MC_TO, MC_YEARS, MC_PATHS = "2026-09-30", 10, 10_000
MC_DRIFT_COLOURS = {"Black-Litterman views": H.GREEN, "Market equilibrium": H.STEEL,
                    "Historical mean": H.AMBER, "Risk-free rate only": H.SILVER}


def montecarlo_exhibits(ds, first_number: int) -> dict:
    """Ten-year projection of the three portfolios under four drift scenarios, in real terms."""
    full = ds.window(C.EST_START, MC_TO)                     # everything known at the projection date
    R, _ = CR.shrunk_corr(full)
    p = MCS.fit(full, R)
    cpi = pd.read_csv(ROOT / "data_cache" / "cpi_monthly.csv", index_col=0, parse_dates=True).iloc[:, 0]
    then = cpi.loc[:cpi.index[-1] - pd.DateOffset(years=30)].iloc[-1]
    infl = float((cpi.iloc[-1] / then) ** (1 / 30) - 1)       # 30-year average CPI inflation
    rf_now = float(ds.rf.loc[:MC_TO].iloc[-1] * 12)
    S, _ = ES.ledoit_wolf_cc(full)
    bl = ES.black_litterman(S, T=len(full))
    drifts = {"Black-Litterman views": bl.posterior + rf_now, "Market equilibrium": bl.prior + rf_now,
              "Historical mean": ES.historical_mean(full), "Risk-free rate only": pd.Series(rf_now, index=full.columns)}
    est = ds.est()
    models = fit(est, float(ds.rf.loc[est.index].mean() * 12), FULL_CAP)
    weights = {"Black-Litterman": models["Black-Litterman"][0], "Historical mean": models["Historical mean"][0],
               "Equal weight (1/N)": pd.Series(1 / len(est.columns), index=est.columns)}
    months = MC_YEARS * 12
    z = MCS.shocks(p, months, MC_PATHS)                       # common random numbers for every scenario
    zg = MCS.shocks(p, months, MC_PATHS, gaussian=True)
    deflate = (1 + infl) ** (np.arange(months + 1) / 12)
    rows, paths = {}, {}
    for pn, w in weights.items():
        for dn, mu in drifts.items():
            res = MCS.simulate(p, w, mu, years=MC_YEARS, start=START_VALUE, z=z)
            rows[(pn, dn)] = {"Expected return": float((w * mu.reindex(w.index)).sum()),
                              **MCS.summarise(res, START_VALUE, MC_YEARS, infl)}
            if pn == "Black-Litterman":
                paths[dn] = res.paths / deflate[:, None]
    res = MCS.simulate(p, weights["Black-Litterman"], drifts["Black-Litterman views"], years=MC_YEARS,
                       start=START_VALUE, z=zg)
    rows[("Black-Litterman", "Black-Litterman views, Gaussian")] = {
        "Expected return": rows[("Black-Litterman", "Black-Litterman views")]["Expected return"],
        **MCS.summarise(res, START_VALUE, MC_YEARS, infl)}
    # short-horizon tails: where the t-copula and a Gaussian simulation differ, if anywhere
    wv = weights["Black-Litterman"].reindex(full.columns).fillna(0).to_numpy()
    mu_m = drifts["Black-Litterman views"].reindex(full.columns).to_numpy() / 12
    tails = {}
    for label, shocks in (("t-copula", z), ("Gaussian", zg)):
        r = np.maximum(mu_m + p.sigma.to_numpy() * shocks, MCS.FLOOR) @ wv            # (months, paths)
        tails[label] = {"P(month < -20%)": float((r < -0.20).mean()),
                        "Monthly 1% quantile": float(np.percentile(r, 1)),
                        "P(12 months < -30%)": float((np.prod(1 + r[:12], axis=0) - 1 < -0.30).mean())}
    pd.DataFrame(tails).to_csv(OUT / f"{first_number + 1}_mc_tail_check.csv")
    table = pd.DataFrame(rows).T
    table.index.names = ["portfolio", "drift"]
    n = [str(first_number + k) for k in range(2)]
    section = "Monte Carlo"
    source = (f"Source: Yahoo Finance prices, Mar 2020 to Sep 2026; FRED CPI; Kenneth French data library; "
              f"author's calculations. {MC_PATHS:,} paths.")
    params = pd.DataFrame({"monthly_vol": p.sigma, "marginal_nu": p.nu})
    params.loc["copula", "marginal_nu"] = p.nu_c
    params.to_csv(OUT / f"{n[0]}_mc_parameters.csv")
    table.to_csv(OUT / f"{n[1]}_mc_scenarios.csv")

    # fan chart, Black-Litterman portfolio
    yrs = np.arange(months + 1) / 12
    focal = paths["Black-Litterman views"]
    qs = {k: np.percentile(focal, k * 100, axis=1) for k in (0.05, 0.25, 0.75, 0.95)}
    meds = {dn: np.median(v, axis=1) for dn, v in paths.items()}
    b, h = rows[("Black-Litterman", "Black-Litterman views")], rows[("Black-Litterman", "Historical mean")]
    fig = X.fan_chart(yrs, qs, meds, colours=MC_DRIFT_COLOURS, start_value=START_VALUE, number=n[0], section=section,
                      dashes={"Market equilibrium": (0, (5, 3)), "Historical mean": (0, (5, 3)),
                              "Risk-free rate only": (0, (1.5, 2.5))}, source=source,
                      title="Black-Litterman portfolio: $500,000 over ten years, in today's money",
                      takeaway=f"Median ${b['Median real end value'] / 1e3:,.0f}k with the model's own drift, "
                               f"${h['Median real end value'] / 1e6:,.1f}m with the historical mean. Same portfolio, same risk model. "
                               f"Inflation {infl:.1%} a year.")
    H.save(fig, OUT / f"{n[0]}_mc_fan")

    # scenario table
    short = {"Black-Litterman": "Black-Litterman", "Historical mean": "Historical mean", "Equal weight (1/N)": "1/N"}
    shown = pd.DataFrame(index=[f"{short[pn]} · {dn}" for pn, dn in table.index])
    shown["Expected\nreturn"] = [f"{v:.1%}" for v in table["Expected return"]]
    shown["P(loss),\nnominal"] = [f"{v:.1%}" for v in table["P(nominal loss)"]]
    shown["P(loss),\nreal"] = [f"{v:.1%}" for v in table["P(real loss)"]]
    shown["Median,\nreal"] = [X._money_fmt()(v, None) for v in table["Median real end value"]]
    shown["5th pct,\nreal"] = [X._money_fmt()(v, None) for v in table["5th percentile real"]]
    shown["Median\ndrawdown"] = [f"{v:.0%}" for v in table["Median worst drawdown"]]
    lo, hi = table["P(nominal loss)"].min(), table["P(nominal loss)"].max()
    g = rows[("Black-Litterman", "Black-Litterman views, Gaussian")]["P(nominal loss)"]
    fig = X.performance_table(shown, number=n[1], focal="", section=section, source=source, first_col=0.38,
                              corner="Portfolio · drift assumed",
                              title="Ten-year outcomes: the drift decides, the tails do not",
                              takeaway=f"Probability of a nominal loss: {lo:.0%} to {hi:.0%}, set by the drift. A Gaussian copula instead "
                                       f"of the t-copula moves it from {b['P(nominal loss)']:.1%} to {g:.1%}.")
    H.save(fig, OUT / f"{n[1]}_mc_scenarios")
    print(f"Monte Carlo: exhibits {n[0]} to {n[1]}; copula nu {p.nu_c:g}, inflation {infl:.2%}, rf {rf_now:.2%}")
    return {"table": table, "params": p, "inflation": infl}


CLUSTER_COLOURS = {"Infrastructure": H.STEEL, "REIT": H.AMBER, "Asia": H.RASPBERRY, "SPY": H.SILVER}


def universe_exhibits(ds) -> None:
    """Exhibit 1, the universe table; exhibit 2, growth of $100 in each cluster against SPY."""
    est = ds.window(C.EST_START, C.EST_END, C.SINGLE_NAMES + C.BENCHMARKS)
    caps = ES.MARKET_CAP_2024
    rows = {}
    for t in C.SINGLE_NAMES + C.BENCHMARKS:
        company, role, listing = C.DESCRIPTIONS[t]
        r = est[t].dropna()                                  # DTCR launched in Oct 2020: shorter history
        since = "" if r.index[0] <= pd.Timestamp(C.EST_START) else "*"
        cluster = "Benchmark" if t == C.BENCHMARK else "Comparison" if t in C.THEMATIC_ETFS else C.TICKER_CLUSTER[t]
        rows[t] = {"Company": company, "Cluster": cluster,
                   "Role in data centres": role, "Listing": listing,
                   "Market cap\nend-2024": f"${caps[t]:,.0f}bn" if t in caps and caps[t] >= 10 else
                                           f"${caps[t]:,.1f}bn" if t in caps else "",
                   "Return\na year": f"{M.cagr(r):+.0%}{since}", "Volatility\na year": f"{r.std() * np.sqrt(12):.0%}{since}"}
    tab = pd.DataFrame(rows).T
    tab.to_csv(OUT / "1_universe.csv")
    fig = X.performance_table(tab, number="1", focal="", section="The universe", corner="Ticker",
                              col_widths=[0.075, 0.2, 0.105, 0.285, 0.1, 0.085, 0.075, 0.075],
                              align=["l", "l", "l", "l", "r", "r", "r"], font=7.2,
                              title="Seventeen stocks in three clusters, and three funds to compare against",
                              takeaway="Market cap on 31 Dec 2024. Return (compound) and volatility: Mar 2020 to Dec 2024, "
                                       "the estimation window.",
                              source="Source: Yahoo Finance total-return prices, Mar 2020 to Dec 2024; author's "
                                     "calculations. *DTCR from its launch, Nov 2020.")
    H.save(fig, OUT / "1_universe")

    daily = {}
    for name in ("Infrastructure", "REIT", "Asia"):
        tick = C.CLUSTERS[name]
        bt = EN.run(ds.prices, pd.Series(1 / len(tick), index=tick), EST_FROM, OOS_TO)
        daily[name] = bt.daily
    first = daily["REIT"].index[0]
    daily["SPY"] = ds.prices["SPY"].loc[first:OOS_TO].dropna()
    end = {k: 100 * v.iloc[-1] / v.iloc[0] for k, v in daily.items()}
    fig = X.wealth_chart(daily, start_value=100, number="2", colours=CLUSTER_COLOURS, section="The universe",
                         log=True, split=(C.EST_END, "Estimation window", "Out of sample"),
                         source="Source: Yahoo Finance total-return prices, Mar 2020 to Sep 2026; author's calculations.",
                         title="Growth of $100 in each cluster, equally weighted",
                         takeaway=f"Log scale. Infrastructure ${end['Infrastructure']:,.0f}, REITs ${end['REIT']:,.0f}, "
                                  f"Asia ${end['Asia']:,.0f}, S&P 500 ${end['SPY']:,.0f} by Sep 2026. Annual rebalancing.")
    H.save(fig, OUT / "2_cluster_growth")
    print("Universe: exhibits 1 and 2", {k: round(v) for k, v in end.items()})


def main() -> None:
    ds = D.build()
    OUT.mkdir(exist_ok=True)
    # Main body in report order: universe, the portfolio, out of sample, outlook.
    universe_exhibits(ds)
    full_portfolio_exhibits(ds, 3)
    oos_exhibits(ds, 9)
    montecarlo_exhibits(ds, 13)
    # Appendix A: the cluster portfolios (supporting analysis); appendix B: correlation structure.
    for k, name in enumerate(("REIT", "Asia", "Infrastructure")):
        cluster_exhibits(ds, name, 1 + 3 * k, prefix="A")
    appendix_exhibits(ds)


if __name__ == "__main__":
    main()
