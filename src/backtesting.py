"""Walk-forward monthly rebalancing backtester with transaction costs."""
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from .optimization import max_sharpe_portfolio, min_var_portfolio, shrink_covariance, portfolio_stats

def available_tickers(window: pd.DataFrame, next_row: pd.Series) -> list:
    """
    Tickers usable at a rebalance: complete history inside the estimation
    window AND a realised return in the holding month.

    The alternative, dropping every row with any missing value across the
    whole universe, lets one late-listing series truncate the sample for
    everyone else. Restricting the investable set per rebalance keeps the
    full history and is still free of look-ahead, since both conditions are
    known at the rebalance date (a listed security has a price next month).
    """
    ok = window.notna().all() & next_row.notna()
    return list(ok[ok].index)


@dataclass
class BacktestConfig:
    lookback_months: int = 24
    max_weight: Optional[float] = 0.15
    method: str = 'max_sharpe'
    cov_method: str = 'ledoit_wolf'
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    turnover_penalty: float = 0.0
    rf_annual: float = 0.045

class WalkForwardBacktest:
    def __init__(self, config=None):
        self.config = config or BacktestConfig()

    def run(self, monthly_returns: pd.DataFrame, tickers: List[str]) -> Dict:
        cfg = self.config
        mr = monthly_returns[tickers].copy()
        dates = mr.index
        port_rets, weights_hist, turnover, costs = [], [], [], []
        prev_w = np.zeros(len(tickers))
        pos = {t: k for k, t in enumerate(tickers)}
        self.n_fallbacks = 0
        for i in range(cfg.lookback_months, len(dates)):
            window = mr.iloc[i - cfg.lookback_months:i]
            avail = available_tickers(window, mr.iloc[i])
            idx = [pos[t] for t in avail]
            sub = window[avail]
            mu = sub.mean().values
            cov = shrink_covariance(sub, method=cfg.cov_method) if cfg.cov_method == 'ledoit_wolf' else sub.cov().values
            w = np.zeros(len(tickers))
            try:
                if cfg.method == 'max_sharpe':
                    w[idx] = max_sharpe_portfolio(mu, cov, rf=cfg.rf_annual, max_weight=cfg.max_weight,
                                                  prev_weights=prev_w[idx], turnover_penalty=cfg.turnover_penalty,
                                                  n_restarts=3, seed=i)
                elif cfg.method == 'min_var':
                    w[idx] = min_var_portfolio(cov, max_weight=cfg.max_weight)
                else:
                    w[idx] = 1.0 / len(idx)
            except Exception:
                w[idx] = 1.0 / len(idx)
                self.n_fallbacks += 1
            actual = mr.iloc[i][tickers].fillna(0.0).values
            gross_ret = np.dot(w, actual)
            to = 0.5 * np.sum(np.abs(w - prev_w))
            tc = to * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000
            port_rets.append(gross_ret - tc)
            weights_hist.append(dict(zip(tickers, w)))
            turnover.append(to)
            costs.append(tc)
            prev_w = w
        idx = dates[cfg.lookback_months:]
        return {
            'returns': pd.Series(port_rets, index=idx, name='Portfolio'),
            'weights': weights_hist,
            'turnover': pd.Series(turnover, index=idx),
            'costs': pd.Series(costs, index=idx),
            'tickers': tickers,
        }

def run_strategy_suite(monthly_returns, tickers, benchmark_col='SPY', lookback=24, rf=0.045):
    strategies = {
        'MaxSharpe_LW': BacktestConfig(lookback_months=lookback, method='max_sharpe',
            cov_method='ledoit_wolf', max_weight=0.15, rf_annual=rf),
        'MaxSharpe_Sample': BacktestConfig(lookback_months=lookback, method='max_sharpe',
            cov_method='sample', max_weight=0.15, rf_annual=rf),
        'MinVar_LW': BacktestConfig(lookback_months=lookback, method='min_var',
            cov_method='ledoit_wolf', max_weight=0.15, rf_annual=rf),
        'EqualWeight': BacktestConfig(lookback_months=lookback, method='equal_weight', rf_annual=rf),
    }
    results = {}
    for name, cfg in strategies.items():
        bt = WalkForwardBacktest(cfg)
        out = bt.run(monthly_returns, tickers)
        results[name] = out['returns']
    if benchmark_col in monthly_returns.columns:
        first_idx = list(results.values())[0].index
        results['SPY'] = monthly_returns[benchmark_col].loc[first_idx]
    return results
