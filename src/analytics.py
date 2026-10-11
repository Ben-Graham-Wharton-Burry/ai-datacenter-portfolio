"""Performance and risk analytics."""
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
from typing import Dict, List

def perf_report(returns: pd.Series, rf_annual=0.045, name='Portfolio') -> Dict:
    rf_m = (1 + rf_annual) ** (1/12) - 1
    T = len(returns)
    if T < 2:
        return {k: np.nan for k in ['CAGR','Volatility','Sharpe','Sortino','Calmar',
                                     'Max DD','VaR 95%','CVaR 95%','Best Mo','Worst Mo',
                                     'Win Rate','Skew','Kurtosis','T']}
    ann_ret = (1 + returns).prod() ** (12/T) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (returns.mean() - rf_m) / returns.std() * np.sqrt(12) if returns.std() > 1e-10 else 0
    downside = returns[returns < 0]
    down_vol = downside.std() * np.sqrt(12) if len(downside) > 1 else 1e-10
    sortino = (returns.mean() - rf_m) * 12 / down_vol
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    max_dd = ((cum - peak) / peak).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    var95 = np.percentile(returns, 5)
    cvar95 = returns[returns <= var95].mean() if (returns <= var95).any() else var95
    return {'CAGR': ann_ret, 'Volatility': ann_vol, 'Sharpe': sharpe, 'Sortino': sortino,
            'Calmar': calmar, 'Max DD': max_dd, 'VaR 95%': var95, 'CVaR 95%': cvar95,
            'Best Mo': returns.max(), 'Worst Mo': returns.min(), 'Win Rate': (returns > 0).mean(),
            'Skew': returns.skew(), 'Kurtosis': returns.kurtosis(), 'T': T}

def compare_strategies(strategy_returns: Dict[str, pd.Series], rf_annual=0.045) -> pd.DataFrame:
    records = {name: perf_report(rets, rf_annual, name) for name, rets in strategy_returns.items()}
    return pd.DataFrame(records).T

def var_analysis(returns: pd.Series, alphas=[0.90, 0.95, 0.99]) -> pd.DataFrame:
    mu, sig = returns.mean(), returns.std()
    rows = []
    for alpha in alphas:
        pct = (1 - alpha) * 100
        hist_var = np.percentile(returns, pct)
        hist_cvar = returns[returns <= hist_var].mean() if (returns <= hist_var).any() else hist_var
        gauss_var = mu + norm.ppf(1 - alpha) * sig
        rows.append({'Confidence': f'{alpha:.0%}', 'Hist VaR': hist_var,
                     'Hist CVaR': hist_cvar, 'Gauss VaR': gauss_var})
    return pd.DataFrame(rows).set_index('Confidence')
