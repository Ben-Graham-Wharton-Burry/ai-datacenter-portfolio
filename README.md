# AI Data Centre Infrastructure Portfolio

Construction and out-of-sample validation of a thematic equity portfolio built around the physical infrastructure behind AI compute: the colocation REITs that house the racks, the firms that power and cool them, and the networking hardware that connects them.

Two notebooks, deliberately separated.

---

## Why two notebooks

`AI_DataCenter_Thesis_Analysis.ipynb` is the research. It builds the universe, decomposes returns into Fama-French factors, optimises portfolios under several objectives, applies Black-Litterman with thematic views, and projects forward by Monte Carlo. It produces encouraging numbers and it contains several biases.

`unified_analysis.ipynb` is the validation. It re-runs the core under a temporal train/test split with an embargo, walk-forward estimation, transaction costs, and formal corrections for the fact that more than one strategy was tried.

Research and validation have different incentives, and running them in one file invites the researcher to adjust the model until the validation passes. Separating them gives the validation a fixed target. **The two notebooks disagree, and that disagreement is the most informative output of the project.**

## What is implemented

**Portfolio construction.** Markowitz tangency and global minimum variance, solved with SLSQP from multiple random starting points because the Sharpe objective is not concave under the budget and bound constraints. Ledoit-Wolf covariance shrinkage. A 15% position cap. Black-Litterman with four thematic views. Effective N reported alongside every weight vector, because a weight vector that looks diversified frequently is not.

**Validation.** Temporal split (2019 to 2022 train, one-month embargo, 2023 to 2026 test), walk-forward expanding-window estimation, 15 bps round-trip costs, and Sharpe decay as the headline diagnostic.

**Significance.** Circular block bootstrap with 10,000 replications and six-month blocks, Probabilistic Sharpe Ratio, Deflated Sharpe Ratio, Harvey-Liu haircut, and a paired block bootstrap against the benchmark.

**Risk.** Historical and parametric VaR and CVaR, marginal contribution to risk via Euler decomposition, and empirical lower tail dependence through the probability integral transform.

## Results

Test period February 2023 to February 2026, net of 15 bps round-trip costs, universe selected on training data only.

| Measure | Result |
|---|---|
| Sharpe ratio | 1.30, bootstrap 95% CI [0.24, 2.74] |
| Probabilistic Sharpe Ratio | 0.97 |
| Deflated Sharpe Ratio, N = 4 | 0.85, not significant |
| Difference against SPY | 0.25, p = 0.50 |
| Equal weight Sharpe ratio | 1.44, against 1.51 for the best optimised variant |

The theme did well over this window, an equal-weighted holding would have captured nearly all of it, and nothing here establishes that the optimisation layer contributed skill. The training-period out-of-sample Sharpe ratio was -0.62, so performance *improved* out of sample, which reflects the 2022 rate shock against the 2023 to 2025 AI rally far more than anything about the strategy. A single split cannot separate the two.

### Research notebook, in sample

In the research notebook's walk-forward, equal weighting had the highest Sharpe ratio (1.64, against 1.61 for maximum Sharpe) and less than half the drawdown (-15% against -34%), with no transaction costs and with the universe-selection bias working in the optimiser's favour. Within-cluster optimisation collapsed three of four clusters to a single name.

## Defects found in review

### Fixed, with a note at the site

**Black-Litterman posterior covariance.** `Sigma_bl` was computed correctly, then passed through pandas `.cov()` before the optimiser, which computes the covariance between the columns of a covariance matrix. Both objects are square matrices of the right size, so nothing raised and the optimiser returned a plausible weight vector.

**PSR kurtosis convention.** The Bailey and Lopez de Prado standard error needs raw kurtosis (normal = 3); the code passed pandas' excess kurtosis (normal = 0), understating the standard error and overstating the PSR.

**DSR benchmark in the wrong units.** sigma_SR was fixed at 1 per month. The real standard error of a 37-month Sharpe estimate is about 0.17, so the expected maximum Sharpe under the null came out at 3.64 for four trials and the DSR was near zero for any strategy. I had first described this as an optimistic simplification. It was the opposite.

**Sharpe decay with a negative denominator.** `1 - SR_test/SR_train` flips sign when the training Sharpe is negative, so a strategy that improved from -0.62 to 1.30 was printed as 308% decay with overfitting risk HIGH. Now reported as undefined.

**The validation notebook had the same universe-selection leak as the research notebook.** It ranked names on full-sample Sharpe, including the test period. It now ranks on training data only.

**Late listings.** DTCR lists in October 2020. In the research notebook a row-wise `dropna` silently moved the whole sample's start to late 2020, so it contains no part of the COVID crash; that is documented rather than changed. In the validation notebook the same series produced NaNs that crashed Ledoit-Wolf, which is why the notebook had never executed. Each rebalance now optimises over names with complete history in its window.

**Unseeded optimiser restarts**, so results changed on every run. **A synthetic-price fallback** that silently replaced real data when the download failed; one copy of the research notebook was executed that way behind a proxy, with nothing in its text to say so. **Factor proxies generated from `np.random.normal`.** All three now either seed, raise, or are removed.

### Documented, not removed

**1. The walk-forward backtest is not out of sample.** Section 4.3 of the research notebook reduces the universe to 15 names by ranking on full-sample Sharpe ratio. Section 6 then runs the walk-forward on that pre-filtered universe. Parameter estimation is genuinely walk-forward; universe selection is not. This is the most consequential issue in the project and the reason the validation notebook exists.

**2. Black-Litterman views are written as absolute returns, one of them below its own prior.** "Vertiv outperforms" is specified as 18% a year against an equilibrium prior of 21.8%, so the bullish view lowers Vertiv's posterior. Only the relative Equinix-against-GDS view behaves as its label says. The prior is also equal-weighted rather than market-cap weighted, which removes the equilibrium argument for the model.

**3. The deflated Sharpe trial count is far too low.** N is set to 4, counting the optimisation variants compared at the end. The honest count includes every choice made along the way: lookback length, rebalancing frequency, weight cap, universe size, shrinkage method, embargo length, block size. Those choices are nested rather than independent, so specifications multiply across layers rather than adding, and the effective search space runs into the thousands. The reported deflation is a lower bound.

The Monte Carlo section carries its own warning. Fifteen-year projections built on seven years of history containing no sustained bear market in the theme, with expected returns taken from a sample that includes the steepest part of an adoption curve, should be read as an illustration of dispersion rather than as a forecast.

## Structure

```
AI_DataCenter_Thesis_Analysis.ipynb   Research: factors, optimisation, BL, Monte Carlo
unified_analysis.ipynb                Validation: split, walk-forward, deflation
src/
├── config.py          Parameters, centralised
├── data.py            Snapshot-first loading, return computation, splits
├── optimization.py    Markowitz, Black-Litterman, Ledoit-Wolf
├── backtesting.py     Walk-forward rebalancing
├── validation.py      Bootstrap, PSR, DSR, overfit diagnostics
├── analytics.py       Performance and risk metrics
└── visualization.py   Charts
data_cache/            Frozen price snapshot
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook unified_analysis.ipynb
```

## References

Black and Litterman (1992). He and Litterman (1999). Ledoit and Wolf (2004). DeMiguel, Garlappi and Uppal (2009). Bailey and Lopez de Prado (2012, 2014). Harvey and Liu (2015). Politis and Romano (1992). Artzner et al. (1999).

## Related

[Backtesting engine](https://github.com/Ben-Graham-Wharton-Burry/backtesting-engine) simulates the resulting weights day by day under realistic execution.

## Author

Benjamin Kristensen, MSc Applied Economics and Finance, Copenhagen Business School.
