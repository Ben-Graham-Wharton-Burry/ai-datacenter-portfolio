"""Publication-quality charts for validation framework."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional
from .config import COLORS, PALETTE

def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'figure.figsize': (14, 7), 'figure.dpi': 120, 'font.size': 11,
                         'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.facecolor': 'white'})

def plot_train_test_equity(train_returns, test_returns, benchmark_returns=None,
                            title='Walk-Forward OOS Returns: Train vs Test'):
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    full = pd.concat([train_returns, test_returns])
    cum = (1 + full).cumprod()
    cum_train = cum.loc[:train_returns.index[-1]]
    ax.plot(cum_train.index, cum_train, lw=2, color=COLORS['navy'], label='Train OOS')
    cum_test = cum.loc[test_returns.index[0]:]
    ax.plot(cum_test.index, cum_test, lw=2.5, color=COLORS['teal'], label='Test OOS')
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(full.index).dropna()
        cum_b = (1 + bench).cumprod()
        ax.plot(cum_b.index, cum_b, lw=1.5, color=COLORS['gray'], ls='--', alpha=0.7, label='S&P 500')
    split = test_returns.index[0]
    ax.axvline(split, color=COLORS['red'], ls='--', lw=2, alpha=0.8)
    ax.text(split, ax.get_ylim()[1]*0.95, '  TEST →', color=COLORS['red'], fontsize=12, fontweight='bold', va='top')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('Cumulative Return ($1 invested)')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.2f'))
    ax.legend(loc='upper left', fontsize=11)

    ax2 = axes[1]
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    dd_train = dd.loc[:train_returns.index[-1]]
    dd_test = dd.loc[test_returns.index[0]:]
    ax2.fill_between(dd_train.index, dd_train, 0, color=COLORS['navy'], alpha=0.3)
    ax2.fill_between(dd_test.index, dd_test, 0, color=COLORS['teal'], alpha=0.4)
    ax2.axvline(split, color=COLORS['red'], ls='--', lw=2, alpha=0.8)
    ax2.set_ylabel('Drawdown')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.tight_layout()
    return fig

def plot_bootstrap_sharpe(boot_result, title='Bootstrap Distribution of Sharpe Ratio', strategy_name='Portfolio'):
    fig, ax = plt.subplots(figsize=(12, 7))
    dist = boot_result['bootstrap_distribution']
    point = boot_result['point_estimate']
    ci_lo, ci_hi = boot_result['ci_lower'], boot_result['ci_upper']
    p_val = boot_result['p_value']
    ax.hist(dist, bins=80, color=COLORS['blue'], alpha=0.6, edgecolor='white', density=True)
    ax.axvspan(ci_lo, ci_hi, alpha=0.15, color=COLORS['teal'], label=f'95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]')
    ax.axvline(point, color=COLORS['navy'], lw=2.5, label=f'Point estimate: {point:.2f}')
    ax.axvline(0, color=COLORS['red'], lw=1.5, ls='--', alpha=0.7, label='Sharpe = 0')
    ax.set_title(f'{title} — {strategy_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annualised Sharpe Ratio')
    ax.set_ylabel('Density')
    sig_text = 'Significant' if p_val < 0.05 else 'Not significant'
    ax.text(0.97, 0.95, f'p-value: {p_val:.4f}\n{sig_text} at 5%', transform=ax.transAxes,
            ha='right', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray'))
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    return fig

def plot_train_test_comparison(diagnostics, title='Train vs Test OOS Performance'):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metrics = ['sharpe', 'cagr', 'max_dd']
    labels = ['Annualised Sharpe', 'CAGR', 'Max Drawdown']
    for ax, metric, label in zip(axes, metrics, labels):
        tv, ev = diagnostics[f'train_{metric}'], diagnostics[f'test_{metric}']
        bars = ax.bar(['Train OOS', 'Test OOS'], [tv, ev],
                       color=[COLORS['navy'], COLORS['teal']], width=0.5, edgecolor='white')
        for bar, val in zip(bars, [tv, ev]):
            fmt = f'{val:.2f}' if 'sharpe' in metric else f'{val:.1%}'
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), fmt,
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.set_title(label, fontweight='bold')
        ax.set_ylim(bottom=min(0, min(tv, ev) * 1.3))
    decay = diagnostics.get('sharpe_decay')
    if decay is not None:
        risk = 'HIGH' if decay > 0.5 else 'MODERATE' if decay > 0.25 else 'LOW'
        fig.text(0.5, 0.02, f'Sharpe Decay: {decay:.1%} ({risk} overfitting risk)',
                 ha='center', fontsize=13, fontweight='bold')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def plot_psr_summary(psr_result, dsr_result=None, title='Statistical Significance'):
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['PSR\n(vs Sharpe=0)']
    values = [psr_result['psr']]
    colors_list = [COLORS['teal'] if psr_result['psr'] > 0.95 else COLORS['gold'] if psr_result['psr'] > 0.80 else COLORS['red']]
    if dsr_result:
        categories.append(f'DSR\n(vs {dsr_result["n_strategies"]} strategies)')
        values.append(dsr_result['psr'])
        colors_list.append(COLORS['teal'] if dsr_result['psr'] > 0.95 else COLORS['gold'] if dsr_result['psr'] > 0.80 else COLORS['red'])
    bars = ax.barh(categories, values, color=colors_list, height=0.4, edgecolor='white')
    ax.axvline(0.95, color=COLORS['red'], ls='--', lw=1.5, alpha=0.7, label='95% threshold')
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y()+bar.get_height()/2, f'{val:.1%}', va='center', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Probability')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    meta = f"Sharpe: {psr_result['sr_hat_annual']:.2f} | Skew: {psr_result['skewness']:.2f} | Kurt: {psr_result['excess_kurtosis']:.2f} | T: {psr_result['T']} mo"
    ax.text(0.5, -0.12, meta, transform=ax.transAxes, ha='center', fontsize=10, style='italic', color='gray')
    plt.tight_layout()
    return fig

def plot_weight_evolution(weights_list, dates, top_n=6, title='Portfolio Weight Evolution'):
    wt_df = pd.DataFrame(weights_list, index=dates)
    top = wt_df.mean().nlargest(top_n).index.tolist()
    plot_wt = wt_df[top].copy()
    plot_wt['Other'] = wt_df[[c for c in wt_df.columns if c not in top]].sum(axis=1)
    fig, ax = plt.subplots(figsize=(16, 8))
    colors_area = [PALETTE[i] for i in range(len(top))] + [COLORS['gray']]
    ax.stackplot(plot_wt.index, plot_wt.T, labels=plot_wt.columns, colors=colors_area, alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Weight')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc='upper left', ncol=4, fontsize=9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig

def plot_strategy_comparison_table(metrics_df, highlight_best=True):
    fig, ax = plt.subplots(figsize=(14, len(metrics_df) * 0.6 + 2))
    ax.axis('off')
    display = metrics_df.copy()
    fmt_pct = ['CAGR', 'Volatility', 'Max DD', 'VaR 95%', 'CVaR 95%', 'Best Mo', 'Worst Mo', 'Win Rate']
    fmt_dec = ['Sharpe', 'Sortino', 'Calmar', 'Skew', 'Kurtosis']
    for c in fmt_pct:
        if c in display.columns:
            display[c] = display[c].map(lambda x: f'{x:.2%}' if pd.notna(x) else '-')
    for c in fmt_dec:
        if c in display.columns:
            display[c] = display[c].map(lambda x: f'{x:.2f}' if pd.notna(x) else '-')
    if 'T' in display.columns:
        display['T'] = display['T'].map(lambda x: f'{int(x)}' if pd.notna(x) else '-')
    table = ax.table(cellText=display.values, colLabels=display.columns, rowLabels=display.index,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor(COLORS['navy'])
            cell.set_text_props(color='white', fontweight='bold')
        elif j == -1:
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(fontweight='bold')
    ax.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig
