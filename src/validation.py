"""
Validation framework: train/test splits, bootstrap significance, PSR, DSR.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
from .optimization import shrink_covariance, max_sharpe_portfolio, min_var_portfolio, portfolio_stats
from .backtesting import BacktestConfig, available_tickers


class WalkForwardValidator:
    """
    Expanding- or rolling-window walk-forward OOS validation,
    with results partitioned into train-period and test-period subsets.
    """
    def __init__(self, min_train_months=24, expanding=True, max_train_months=None,
                 cov_method='ledoit_wolf', max_weight=0.15, rf_annual=0.045,
                 transaction_cost_bps=10.0, slippage_bps=5.0):
        self.min_train = min_train_months
        self.expanding = expanding
        self.max_train = max_train_months
        self.cov_method = cov_method
        self.max_weight = max_weight
        self.rf = rf_annual
        self.tc_bps = transaction_cost_bps
        self.slip_bps = slippage_bps

    def run(self, monthly_returns, tickers, train_end, test_start):
        mr = monthly_returns[tickers].copy()
        dates = mr.index
        train_cutoff = pd.Timestamp(train_end)
        test_cutoff  = pd.Timestamp(test_start)

        oos_rets, oos_dates, weights_hist, turnover_hist = [], [], [], []
        prev_w = np.zeros(len(tickers))
        pos = {t: k for k, t in enumerate(tickers)}
        self.n_fallbacks = 0

        for i in range(self.min_train, len(dates)):
            start_idx = 0 if self.expanding else max(0, i - (self.max_train or self.min_train))
            window = mr.iloc[start_idx:i]
            avail = available_tickers(window, mr.iloc[i])
            idx = [pos[t] for t in avail]
            sub = window[avail]
            mu = sub.mean().values
            cov = shrink_covariance(sub, method=self.cov_method)

            w = np.zeros(len(tickers))
            try:
                w[idx] = max_sharpe_portfolio(mu, cov, rf=self.rf, max_weight=self.max_weight,
                                              prev_weights=prev_w[idx], n_restarts=3, seed=i)
            except Exception:
                # Counted and reported, not hidden
                w[idx] = 1.0 / len(idx)
                self.n_fallbacks += 1

            actual = mr.iloc[i].fillna(0.0).values
            gross_ret = w @ actual
            to = 0.5 * np.sum(np.abs(w - prev_w))
            tc = to * (self.tc_bps + self.slip_bps) / 10_000

            oos_rets.append(gross_ret - tc)
            oos_dates.append(dates[i])
            weights_hist.append(dict(zip(tickers, w)))
            turnover_hist.append(to)
            prev_w = w

        full = pd.Series(oos_rets, index=pd.DatetimeIndex(oos_dates), name='OOS_Return')
        to_series = pd.Series(turnover_hist, index=full.index, name='Turnover')

        mask_train = full.index <= train_cutoff
        mask_test  = full.index >= test_cutoff

        result = {
            'full_oos_returns':  full,
            'oos_returns_train': full[mask_train],
            'oos_returns_test':  full[mask_test],
            'weights_train': [w for w, m in zip(weights_hist, mask_train) if m],
            'weights_test':  [w for w, m in zip(weights_hist, mask_test) if m],
            'turnover_train': to_series[mask_train],
            'turnover_test':  to_series[mask_test],
        }
        result['diagnostics'] = self._diagnostics(result)
        result['diagnostics']['optimiser_fallbacks'] = self.n_fallbacks
        return result

    def _diagnostics(self, result):
        rf_m = (1 + self.rf) ** (1/12) - 1
        def _sharpe(r):
            return (r.mean() - rf_m) / r.std() * np.sqrt(12) if len(r) > 2 and r.std() > 1e-10 else 0.0
        def _cagr(r):
            return (1 + r).prod() ** (12 / len(r)) - 1 if len(r) > 1 else 0.0
        def _max_dd(r):
            cum = (1 + r).cumprod(); peak = cum.expanding().max()
            return ((cum - peak) / peak).min()

        tr, te = result['oos_returns_train'], result['oos_returns_test']
        d = {
            'train_sharpe': _sharpe(tr), 'test_sharpe': _sharpe(te),
            'train_cagr': _cagr(tr), 'test_cagr': _cagr(te),
            'train_max_dd': _max_dd(tr), 'test_max_dd': _max_dd(te),
            'train_months': len(tr), 'test_months': len(te),
            'train_turnover_avg': result['turnover_train'].mean(),
            'test_turnover_avg': result['turnover_test'].mean(),
            'sharpe_decay': None,
        }
        # The ratio 1 - SR_test/SR_train is only interpretable when SR_train > 0.
        # With a negative denominator it flips sign: a strategy that did far
        # BETTER out of sample would be reported as catastrophic decay.
        if d['train_sharpe'] > 0:
            d['sharpe_decay'] = 1 - d['test_sharpe'] / d['train_sharpe']
        return d


class SignificanceTests:
    """Bootstrap CI, PSR, DSR, and paired tests."""

    @staticmethod
    def circular_block_bootstrap_sharpe(returns, n_bootstrap=10000, block_size=6,
                                         rf_annual=0.045, confidence=0.95, seed=42):
        rng = np.random.RandomState(seed)
        r = returns.values
        T = len(r)
        rf_m = (1 + rf_annual) ** (1/12) - 1
        n_blocks = int(np.ceil(T / block_size))
        boot_sharpes = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            starts = rng.randint(0, T, size=n_blocks)
            indices = np.concatenate([np.arange(s, s + block_size) % T for s in starts])[:T]
            br = r[indices]
            mu_b = br.mean() - rf_m
            sig_b = br.std()
            boot_sharpes[b] = (mu_b / sig_b * np.sqrt(12)) if sig_b > 1e-10 else 0.0

        mu_hat = r.mean() - rf_m
        sig_hat = r.std()
        point_est = (mu_hat / sig_hat * np.sqrt(12)) if sig_hat > 1e-10 else 0.0
        alpha = 1 - confidence
        ci_lo = np.percentile(boot_sharpes, 100 * alpha / 2)
        ci_hi = np.percentile(boot_sharpes, 100 * (1 - alpha / 2))
        p_value = np.mean(boot_sharpes <= 0)

        return {'point_estimate': point_est, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
                'se': np.std(boot_sharpes), 'p_value': p_value,
                'bootstrap_distribution': boot_sharpes}

    @staticmethod
    def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0, rf_annual=0.045):
        r = returns.values
        T = len(r)
        rf_m = (1 + rf_annual) ** (1/12) - 1
        excess = r - rf_m
        sr_hat = excess.mean() / excess.std() if excess.std() > 1e-10 else 0.0
        sr_bench_m = sr_benchmark / np.sqrt(12)
        gamma3 = pd.Series(r).skew()
        # Bailey and Lopez de Prado's standard error requires RAW kurtosis,
        # where a normal distribution gives 3. pandas .kurtosis() returns
        # Fisher (excess) kurtosis, where a normal gives 0. Passing the pandas
        # value directly makes (gamma4 - 1)/4 evaluate to -0.25 instead of
        # +0.5 under normality, which understates the standard error and
        # overstates the PSR by a margin that grows with sr_hat^2.
        excess_kurt = pd.Series(r).kurtosis()
        gamma4 = excess_kurt + 3.0
        se_sr = np.sqrt((1 - gamma3 * sr_hat + (gamma4 - 1) / 4 * sr_hat**2) / (T - 1))

        if se_sr < 1e-10:
            psr = 1.0 if sr_hat > sr_bench_m else 0.0
            z_stat = np.inf if sr_hat > sr_bench_m else -np.inf
        else:
            z_stat = (sr_hat - sr_bench_m) / se_sr
            psr = stats.norm.cdf(z_stat)

        return {'psr': psr, 'z_stat': z_stat, 'sr_hat_monthly': sr_hat,
                'sr_hat_annual': sr_hat * np.sqrt(12), 'se_sr': se_sr,
                'skewness': gamma3, 'excess_kurtosis': excess_kurt,
                'raw_kurtosis': gamma4, 'T': T,
                'significant': psr > 0.95}

    @staticmethod
    def deflated_sharpe_ratio(returns, n_strategies_tested, rf_annual=0.045,
                              trial_sharpes_monthly=None):
        """
        Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014).

        The benchmark is the expected maximum Sharpe ratio across N trials with
        no skill, E[max SR] = sigma_SR * [(1-g) Z(1-1/N) + g Z(1-1/(N e))].

        sigma_SR is the dispersion of Sharpe ratio ESTIMATES across trials. If
        the trial Sharpe ratios are supplied it is their sample standard
        deviation; otherwise it is the standard error of a Sharpe estimate under
        the null, sqrt(1/(T-1)) in per-period units.

        An earlier version set sigma_SR = 1 in monthly units. For a 37-month
        sample the true standard error is about 0.17, so that inflated the
        benchmark roughly six-fold and drove the DSR to zero almost regardless
        of the strategy. It was too harsh, not too lenient.
        """
        N = max(n_strategies_tested, 1)
        T = len(returns)
        gamma_em = 0.5772156649
        if trial_sharpes_monthly is not None and len(trial_sharpes_monthly) > 1:
            sigma_sr = float(np.std(trial_sharpes_monthly, ddof=1))
        else:
            sigma_sr = np.sqrt(1.0 / (T - 1))
        if N > 1:
            z = ((1 - gamma_em) * stats.norm.ppf(1 - 1/N) +
                 gamma_em * stats.norm.ppf(1 - 1/(N * np.e)))
            sr_expected_max_annual = sigma_sr * z * np.sqrt(12)
        else:
            sr_expected_max_annual = 0.0
        result = SignificanceTests.probabilistic_sharpe_ratio(
            returns, sr_benchmark=sr_expected_max_annual, rf_annual=rf_annual)
        result['n_strategies'] = N
        result['sr_benchmark_annual'] = sr_expected_max_annual
        result['sigma_sr_monthly'] = sigma_sr
        return result

    @staticmethod
    def paired_bootstrap_test(returns_a, returns_b, n_bootstrap=10000, block_size=6, seed=42,
                              rf_annual=0.045):
        rng = np.random.RandomState(seed)
        common = returns_a.index.intersection(returns_b.index)
        # Excess returns, so these Sharpe ratios match the rest of the project
        rf_m = (1 + rf_annual) ** (1/12) - 1
        ra = returns_a.loc[common].values - rf_m
        rb = returns_b.loc[common].values - rf_m
        T = len(ra)
        n_blocks = int(np.ceil(T / block_size))
        def _sharpe(r):
            return r.mean() / r.std() * np.sqrt(12) if r.std() > 1e-10 else 0.0
        sr_a, sr_b = _sharpe(ra), _sharpe(rb)
        obs_diff = sr_a - sr_b
        boot_diffs = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            starts = rng.randint(0, T, size=n_blocks)
            indices = np.concatenate([np.arange(s, s + block_size) % T for s in starts])[:T]
            boot_diffs[b] = _sharpe(ra[indices]) - _sharpe(rb[indices])
        p_value = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
        return {'sharpe_a': sr_a, 'sharpe_b': sr_b, 'observed_diff': obs_diff,
                'p_value': p_value, 'ci_lower': np.percentile(boot_diffs, 2.5),
                'ci_upper': np.percentile(boot_diffs, 97.5),
                'significant_5pct': p_value < 0.05}


class OverfitDiagnostics:
    @staticmethod
    def sharpe_ratio_haircuts(observed_sharpe, n_months, n_strategies=1):
        if n_strategies <= 1:
            haircut = 0.0
        else:
            haircut = np.sqrt(2 * np.log(n_strategies)) / np.sqrt(n_months / 12)
        adjusted = max(0, observed_sharpe - haircut)
        return {'observed': observed_sharpe, 'haircut': haircut, 'adjusted': adjusted,
                'haircut_pct': haircut / observed_sharpe * 100 if observed_sharpe > 0 else 0,
                'n_months': n_months, 'n_strategies': n_strategies}
