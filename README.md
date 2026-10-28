# AI Data Centres: An Investment Thesis, Rebuilt

How should an investor hold the AI data-centre build-out? This project splits seventeen listed companies into three clusters (the REITs that own the buildings, two Chinese operators, and ten suppliers of power, cooling and networking equipment), builds portfolios from them with two methods, tests those portfolios on 21 months of data they never saw, and projects them ten years ahead.

**[Read the full note (PDF, 30 pages)](AI_Data_Centres_Investment_Thesis.pdf)**

Every number, table and chart in the note is produced by the code in this repository, from one data snapshot, in one run.

## Our view

The thesis survives in a narrower form than the first version claimed. The theme was real and the infrastructure names delivered it. What the optimiser adds on top of a simple equal-weighted basket is small and, on the evidence available, cannot be distinguished from luck.

| Key numbers | |
|---|---|
| $100 in the infrastructure suppliers, Mar 2020 to Sep 2026 | $750, against $283 in the S&P 500 |
| Black-Litterman portfolio, 15% position cap | 13 names, effective number of positions 10.1 |
| $500,000 out of sample, Jan 2025 to Sep 2026 | $778k Black-Litterman, $763k equal weight, $663k S&P 500 |
| Out-of-sample Sharpe ratio, 95% interval | Black-Litterman 0.93 (−0.57 to 2.44), S&P 500 1.04 |
| Volatility, forecast against realised | Black-Litterman 22% vs 26.5%; historical mean 25% vs 40% |
| Ten years ahead on Black-Litterman views, today's money | Median $822k from $500k; 15% chance of a nominal loss |

![Growth of $100 in each cluster](exhibits/2_cluster_growth.png)

## What the analysis found

* **The suppliers carried the theme.** Infrastructure returned 36% a year from March 2020 to September 2026, the REITs 9% and the S&P 500 17%. The Chinese operators lost about half their value.
* **Optimising on historical returns looks best on the data it was fitted to.** It concentrates in the past winners (error maximisation, in Michaud's phrase). Black-Litterman, anchored to market capitalisation, holds a diversified portfolio with modest expectations.
* **Out of sample, no method can be told apart from the market.** Every Sharpe ratio carries a 95% interval of about ±1.5 over 21 months. Black-Litterman ended level with an equally weighted portfolio of the same stocks.
* **Risk is the forecast the data can test.** Black-Litterman's volatility forecast landed inside its interval. The historical-mean forecast understated realised volatility by more than a third.
* **Ten years ahead, the expected-return assumption decides almost everything.** The same portfolio has a 0% or a 46% chance of losing money depending on that one input. Fat tails barely move the ten-year numbers.

![Which forecasts can 21 months test?](exhibits/12_oos_forecast_check.png)

## Method in brief

| Step | Choice | Why |
|---|---|---|
| Windows | Estimate Mar 2020 to Dec 2024 (58 months), test Jan 2025 to Sep 2026 (21 months) | One window for every portfolio, and a test set the weights never saw |
| Covariance | Ledoit-Wolf shrinkage to constant correlation (intensity 0.24) | 17 stocks on 58 months is a noisy sample matrix |
| Expected returns | Historical mean against Black-Litterman with a market-cap prior and four views | The input that matters most and is estimated worst |
| Optimisation | Long-only maximum Sharpe in convex form, 15% position cap | A single optimum, no dependence on starting values |
| Out of sample | Design fixed in advance; benchmarks S&P 500, two thematic ETFs and 1/N | 1/N is the test of whether optimisation adds anything |
| Inference | Lo (2002) intervals on Sharpe ratios; sampling intervals on realised return and volatility | Shows what 21 months can and cannot decide |
| Simulation | Student-t marginals joined by a Student-t copula, 10,000 paths, common random numbers, four drift assumptions | Separates the effect of fat tails from the effect of the return assumption |
| Robustness | Hierarchical clustering with linkage fixed in advance, scored by adjusted Rand index | Tests whether the business-line clusters exist in the returns |

The note also covers the risks to the thesis, including the hindsight in choosing the universe, which flatters every in-sample result.

## Reproducing the results

```bash
pip install -r requirements.txt

python scripts/refresh_data.py       # downloads a snapshot into data_cache/
python scripts/build_exhibits.py     # rebuilds every exhibit and CSV in exhibits/
python -m pytest -q                  # 35 tests

# Optional: typeset the PDF
python -m playwright install chromium
python scripts/render_pdf.py
```

Prices come from Yahoo Finance, the risk-free rate from the Kenneth French data library and inflation from FRED. **No market data is redistributed in this repository**: `refresh_data.py` downloads your own copy. If a download is blocked (corporate networks often block the French library), the script saves a diagnostic file and reads manually downloaded copies from `data_cache/manual/` instead. Without a snapshot the four data tests skip and the rest run.

The note uses a snapshot taken on 26 October 2026, with complete months to 30 September 2026. Two builds from the same snapshot produce byte-identical CSVs. A later snapshot will give slightly different numbers, because the providers revise their histories.

## Repository layout

```
AI_Data_Centres_Investment_Thesis.pdf   the note
docs/manuscript.md                      its text; render_pdf.py typesets it with the exhibits
exhibits/                               every exhibit as PNG and vector PDF, with CSVs of the numbers
scripts/
  refresh_data.py                       data snapshot: prices, factors, CPI, market caps
  build_exhibits.py                     the whole analysis, in report order
  render_pdf.py                         markdown to PDF
src/
  config.py                             universe, clusters, windows, views
  dataset.py                            loading and cleaning, including the Vertiv SPAC fix
  estimators.py                         sample, Ledoit-Wolf, Black-Litterman
  optimiser.py                          convex maximum Sharpe with a position cap
  engine.py                             daily backtest with annual rebalancing and costs
  metrics.py                            performance table and Sharpe intervals
  correlation.py                        hierarchical clustering and agreement tests
  montecarlo.py                         t-copula simulation
  market_caps.py                        market caps from share counts, with ADR handling
  exhibits.py, house_style.py           charts and tables
tests/                                  unit tests, including hand-checked Sharpe intervals
```

## Version history

The first version (tag `v1`) took its optimisations, charts and simulations from Portfolio Visualizer, scored portfolios on the data they were fitted to, mixed estimation windows of different lengths, and reported a 38% expected return with a 0% chance of loss. This version replaces all of it with our own code. Appendix D of the note lists the changes.

## Related

[Backtesting engine](https://github.com/Ben-Graham-Wharton-Burry/backtesting-engine): an event-driven backtester with pluggable strategies, transaction costs and an audit trail.

---

High_Magus_of_Risk. A personal research project, not investment advice.
