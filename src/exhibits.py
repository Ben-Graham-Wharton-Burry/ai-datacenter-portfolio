"""
Exhibits for the document. Each function draws one exhibit in the house
style and returns the figure; build_exhibits.py saves them all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C, optimiser as OP
from . import house_style as H

SOURCE_EST = "Source: Yahoo Finance total-return prices, Mar 2020 to Dec 2024; author's calculations."


def weights_comparison(w_focal: pd.Series, w_ref: pd.Series, *, number: str, title: str,
                       focal_label: str, ref_label: str, cap: float | None = None,
                       section: str = "Portfolio construction", source: str = SOURCE_EST,
                       ref_colour: str = H.STEEL, note: str = ""):
    """Paired horizontal bars of two weight vectors, ticker and cluster on the left."""
    W = pd.DataFrame({ref_label: w_ref, focal_label: w_focal}).fillna(0.0)
    W = W.loc[W.max(axis=1) > 0.001].sort_values([focal_label, ref_label])
    fig, ax = H.figure(5.4)
    y, h = np.arange(len(W)), 0.38
    for k, (col, colour) in enumerate(((ref_label, ref_colour), (focal_label, H.GREEN))):
        vals = W[col].to_numpy() * 100
        off = (k - 0.5) * (h + 0.04)
        ax.barh(y + off, vals, height=h, color=colour, label=col, zorder=3)
        for yi, v in zip(y, vals):
            if v > 0.05:
                ax.text(v + 0.25, yi + off, f"{v:.1f}", va="center", fontsize=7, color=H.INK2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(W.index, fontweight="semibold")
    clusters = [C.TICKER_CLUSTER[t] for t in W.index]
    if len(set(clusters)) > 1:                       # only worth a label when the clusters differ
        H.side_labels(ax, y, clusters)
    top = W.to_numpy().max() * 100
    if cap is not None:
        ax.axvline(cap * 100, color=H.INK3, lw=0.8, ls=(0, (3, 3)), zorder=2)
        ax.text(cap * 100 + 0.2, len(W) - 0.4, f"{cap:.0%} cap", fontsize=7, color=H.INK3, va="bottom")
        top = max(top, cap * 100)
    ax.set_xlim(0, top * 1.17)
    ax.set_xlabel("Weight, % of portfolio")
    H.grid(ax, "x")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right", frameon=False)
    takeaway = (f"{ref_label}: effective N {OP.effective_n(W[ref_label]):.1f}.  "
                f"{focal_label}: effective N {OP.effective_n(W[focal_label]):.1f}." + (f"  {note}" if note else ""))
    H.frame(fig, number, section, title, takeaway, source, left=0.27 if len(set(clusters)) > 1 else 0.14,
            right=0.96)
    return fig


def performance_table(table: pd.DataFrame, *, number: str, title: str, takeaway: str,
                      focal: str, section: str = "Portfolio construction", source: str = SOURCE_EST,
                      rows: list[str] | None = None, first_col: float = 0.34, corner: str = "",
                      col_widths: list[float] | None = None, align: list[str] | None = None, font: float = 8.0):
    """A formatted summary table drawn as an exhibit; the focal column is set in green."""
    t = table.loc[rows] if rows else table
    n_rows, n_cols = t.shape
    hr = 1.75 if any("\n" in str(c) for c in t.columns) else 1.0   # header band, in row heights
    units = n_rows + hr
    fig, ax = H.figure(0.62 + 0.235 * units)
    ax.axis("off")
    col_w = col_widths or [first_col] + [(1 - first_col) / n_cols] * n_cols
    xs = np.cumsum([0] + col_w[:-1])
    align = align or ["r"] * n_cols
    pos = lambda j: xs[j] + 0.006 if align[j - 1] == "l" else xs[j] + col_w[j] - 0.01   # j >= 1
    ha = lambda j: "left" if align[j - 1] == "l" else "right"
    row_h = 1.0 / units
    rule_y = 1.0 - hr * row_h
    for j, c in enumerate([corner] + list(t.columns)):
        colour = H.GREEN_DARK if c == focal else H.INK2
        ax.text(pos(j) if j else 0, rule_y + row_h * 0.18, c, ha=ha(j) if j else "left",
                va="bottom", fontsize=8.2, fontweight="semibold", color=colour, transform=ax.transAxes,
                linespacing=1.15)
    ax.plot([0, 1], [rule_y, rule_y], color=H.GREEN_DARK, lw=0.9, transform=ax.transAxes)
    for i, (name, row) in enumerate(t.iterrows()):
        yc = rule_y - row_h * (i + 0.5)
        if i % 2 == 1:
            ax.add_patch(plt_rect(0, yc - row_h / 2, 1, row_h))
        ax.text(0.005, yc, name, ha="left", va="center", fontsize=font, color=H.INK, transform=ax.transAxes,
                fontweight="semibold" if col_widths else "normal")
        for j, c in enumerate(t.columns, start=1):
            ax.text(pos(j), yc, row[c], ha=ha(j), va="center", fontsize=font,
                    color=H.GREEN_DARK if c == focal else H.INK, transform=ax.transAxes,
                    fontweight="semibold" if c == focal else "normal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    head = 0.62 / (0.62 + 0.235 * units)
    H.frame(fig, number, section, title, takeaway, source, top=1 - head * 0.98 - 0.02, bottom=0.07,
            left=0.03, right=0.97)
    _fit_frame(fig, head)
    return fig


def plt_rect(x, y, w, h):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, facecolor="#f3f5f4", edgecolor="none", zorder=0)


def _fit_frame(fig, head_share):
    """The frame helper places text at fixed heights; on a short figure, move it to fit the header band."""
    texts = fig.texts
    lines = [a for a in fig.artists]
    # kicker, title, takeaway, source in that order
    kicker, title, take, src = texts[-4:]
    top_band = 1 - head_share
    kicker.set_y(1 - head_share * 0.16)
    title.set_y(1 - head_share * 0.46)
    take.set_y(1 - head_share * 0.74)
    for ln in lines[-1:]:
        ln.set_ydata([top_band + head_share * 0.04] * 2)
    src.set_y(0.012)


def wealth_chart(paths: dict, *, start_value: float, number: str, title: str, takeaway: str,
                 colours: dict, section: str = "Portfolio construction", source: str = SOURCE_EST,
                 window: str = "", dashes: dict | None = None, log: bool = False,
                 split: tuple | None = None):
    """Growth of ``start_value`` for each daily value path, end values labelled at the right edge.

    ``log`` draws the value axis on a log scale (equal distances are equal percentage moves);
    ``split`` = (date, left label, right label) shades everything after the date.
    """
    fig, ax = H.figure(4.6)
    if log:
        ax.set_yscale("log")
    ends = {}
    for name, v in paths.items():
        w = v / v.iloc[0] * start_value
        lw = 2.0 if colours[name] == H.GREEN else 1.4
        ax.plot(w.index, w.values, color=colours[name], lw=lw, label=name, zorder=3 if lw > 1.5 else 2,
                ls=(dashes or {}).get(name, "-"))
        ends[name] = (w.index[-1], w.iloc[-1])
    # Direct labels at the right edge, nudged apart so they never overlap (in log space on a log axis)
    import matplotlib.dates as mdates
    ymin, ymax = ax.get_ylim()
    f, finv = (np.log10, lambda v: 10 ** v) if log else ((lambda v: v), (lambda v: v))
    gap = (f(ymax) - f(ymin)) * 0.075
    placed = []
    for name in sorted(ends, key=lambda k: ends[k][1]):
        x, y = ends[name]
        y_lab = max(f(y), placed[-1] + gap) if placed else f(y)
        placed.append(y_lab)
        val = f"${y / 1e6:,.2f}m" if y >= 1e6 else f"${y / 1e3:,.0f}k" if y >= 1e3 else f"${y:,.0f}"
        ax.text(x + pd.Timedelta(days=12), finv(y_lab), f"{name}\n{val}", va="center", fontsize=7.6,
                color=colours[name], fontweight="semibold", clip_on=False)
    if split is not None:
        when, left_lab, right_lab = split
        ax.axvspan(pd.Timestamp(when), max(v.index[-1] for v in paths.values()),
                   color=H.SILVER_LIGHT, alpha=0.25, lw=0, zorder=0)
        ax.axvline(pd.Timestamp(when), color=H.INK3, lw=0.8, zorder=1)
        ytxt = finv(f(ymax) - (f(ymax) - f(ymin)) * 0.03)
        ax.text(pd.Timestamp(when) - pd.Timedelta(days=20), ytxt, left_lab, ha="right", va="top", fontsize=7, color=H.INK2)
        ax.text(pd.Timestamp(when) + pd.Timedelta(days=20), ytxt, right_lab, ha="left", va="top", fontsize=7, color=H.INK2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axhline(start_value, color=H.INK3, lw=0.7, ls=(0, (2, 3)), zorder=1)
    ax.yaxis.set_major_formatter(_money_fmt())
    if log:
        from matplotlib.ticker import FixedLocator, NullLocator
        lo, hi = ax.get_ylim()
        ax.yaxis.set_major_locator(FixedLocator([v * m for v in (1, 10, 100, 1e3, 1e4, 1e5, 1e6) for m in (1, 2, 5)
                                                 if lo <= v * m <= hi]))
        ax.yaxis.set_minor_locator(NullLocator())
    H.grid(ax, "y")
    ax.spines["left"].set_visible(False)
    ax.margins(x=0)
    H.frame(fig, number, section, title, takeaway, source, left=0.09, right=0.8)
    return fig


def _money_fmt():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: f"${v / 1e6:,.1f}m" if v >= 1e6 else f"${v / 1e3:,.0f}k" if v >= 1e3
                         else f"${v:,.0f}")


def cluster_mix(mix: pd.DataFrame, *, number: str, title: str, takeaway: str, colours: dict,
                section: str = "Portfolio construction", source: str = SOURCE_EST):
    """Grouped horizontal bars: share of the portfolio in each cluster, one bar per model.

    ``mix`` has clusters as rows and models as columns; the column order is the bar order
    from top to bottom within each group, and the row order is the group order from the top.
    """
    clusters, models = list(mix.index), list(mix.columns)
    fig, ax = H.figure(4.3)
    n, h = len(models), 0.8 / len(models)
    y = np.arange(len(clusters))[::-1]                 # first cluster at the top
    for k, m in enumerate(models):
        off = ((n - 1) / 2 - k) * h
        vals = mix[m].to_numpy() * 100
        ax.barh(y + off, vals, height=h * 0.9, color=colours[m], label=m, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.8, yi + off, f"{v:.1f}", va="center", fontsize=7, color=H.INK2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(clusters, fontweight="semibold")
    ax.set_xlim(0, mix.to_numpy().max() * 100 * 1.15)
    ax.set_xlabel("Share of portfolio, %")
    H.grid(ax, "x")
    ax.legend(loc="lower right", frameon=False)
    H.frame(fig, number, section, title, takeaway, source, left=0.16, right=0.96, bottom=0.14)
    return fig


def frontier_chart(frontiers: dict, optimal: dict, assets: pd.DataFrame, *, colours: dict, number: str,
                   title: str, takeaway: str, benchmark: tuple | None = None,
                   label_at: dict | None = None,
                   section: str = "Portfolio construction", source: str = SOURCE_EST):
    """Ex-ante efficient frontiers, each model's maximum-Sharpe point, and the single stocks.

    frontiers: name -> DataFrame with columns vol, ret (annual, decimals)
    optimal:   name -> (vol, ret) of the chosen portfolio
    assets:    DataFrame indexed by ticker with columns vol, ret
    benchmark: (name, vol, ret) or None
    label_at:  ticker or model name -> (dx, dy, ha) in points, to move a crowded label
    """
    label_at = label_at or {}
    fig, ax = H.figure(5.0)
    ax.scatter(assets["vol"] * 100, assets["ret"] * 100, s=16, facecolor=H.PAPER, edgecolor=H.INK3,
               lw=0.8, zorder=3)
    for t, r in assets.iterrows():
        dx, dy, ha = label_at.get(t, (4, 1, "left"))
        ax.annotate(t, (r["vol"] * 100, r["ret"] * 100), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.4, color=H.INK3, va="center", ha=ha)
    for name, f in frontiers.items():
        focal = colours[name] == H.GREEN
        ax.plot(f["vol"] * 100, f["ret"] * 100, color=colours[name], lw=2.0 if focal else 1.6,
                zorder=4 if focal else 3)
    for name, (v, r) in optimal.items():
        ax.scatter(v * 100, r * 100, s=58, marker="D", color=colours[name], edgecolor=H.PAPER, lw=1.2, zorder=6)
        dx, dy, ha = label_at.get(name, (-8, 0, "right"))
        ax.annotate(f"{name}\nmaximum Sharpe", (v * 100, r * 100), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=7.4, fontweight="semibold", color=colours[name])
    if benchmark is not None:
        name, v, r = benchmark
        ax.scatter(v * 100, r * 100, s=46, marker="s", color=colours[name], edgecolor=H.PAPER, lw=1.2, zorder=6)
        dx, dy, ha = label_at.get(name, (-7, 0, "right"))
        ax.annotate(name, (v * 100, r * 100), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center", fontsize=7.4, fontweight="semibold", color=colours[name])
    ax.axhline(0, color=H.INK3, lw=0.6, zorder=1)
    ax.set_xlim(0, None)
    ax.set_xlabel("Volatility, % a year")
    ax.set_ylabel("Expected return, % a year")
    H.grid(ax, "y")
    ax.xaxis.grid(True, color=H.GRID, lw=0.6, zorder=0)
    H.frame(fig, number, section, title, takeaway, source, left=0.09, right=0.97)
    return fig


def _diverging():
    """Blue (negative) to a neutral grey to green (positive); validated for colour-blind readers."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("house_div", [H.STEEL, "#eef0ef", H.GREEN])
    cmap.set_bad(H.PAPER)
    return cmap


def dendrogram_chart(Z, labels: list[str], clusters: dict, cut: float, *, number: str, title: str,
                     takeaway: str, section: str = "Appendix", source: str = SOURCE_EST):
    """Horizontal dendrogram: leaves on the left with each stock's cluster, the k-group cut dashed."""
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = H.figure(5.0)
    # orientation="right": leaves at distance 0 on the left next to their labels, the root to the right
    out = dendrogram(Z, orientation="right", labels=labels, ax=ax, link_color_func=lambda k: H.INK2,
                     leaf_font_size=8)
    ax.yaxis.tick_left()
    for t in ax.get_yticklabels():
        t.set_fontweight("semibold")
        t.set_color(H.INK)
    ys = ax.get_yticks()
    H.side_labels(ax, ys, [clusters[t] for t in out["ivl"]], x=-0.095)
    ax.axvline(cut, color=H.GREEN, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(cut, ys[-1] + 6, "3-group cut", color=H.GREEN, fontsize=7.4, fontweight="semibold",
            ha="center", va="bottom")
    ax.set_xlabel("Mantegna distance √(½(1 − ρ)) at which branches join: further left means more alike")
    ax.set_xlim(0, max(Z[:, 2]) * 1.04)
    H.grid(ax, "x")
    ax.spines["left"].set_visible(False)
    H.frame(fig, number, section, title, takeaway, source, left=0.25, right=0.97)
    return fig


def correlation_panels(panels: list, order: list[str], *, number: str, title: str, takeaway: str,
                       section: str = "Appendix", source: str = SOURCE_EST):
    """Two correlation heatmaps side by side, same stock order, each with its own scale bar.

    panels: [(subtitle, matrix, vlim, scale label), ...]; the colour scale runs from -vlim to +vlim.
    """
    fig, axes = H.figure(5.3, ncols=len(panels))
    cmap = _diverging()
    for ax, (sub, M, vlim, cbar_label) in zip(axes, panels):
        A = M.loc[order, order].to_numpy(dtype=float).copy()
        np.fill_diagonal(A, np.nan)                     # the diagonal is 1 by definition: leave it blank
        mesh = ax.pcolormesh(A, cmap=cmap, vmin=-vlim, vmax=vlim, edgecolors=H.PAPER, linewidth=0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(order)) + 0.5)
        ax.set_yticks(np.arange(len(order)) + 0.5)
        ax.set_xticklabels(order, rotation=90, fontsize=6.4)
        ax.set_yticklabels(order if ax is axes[0] else [], fontsize=6.4)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(sub, fontsize=8.6, color=H.INK, fontweight="semibold", pad=34, loc="left")
        cax = ax.inset_axes([0.0, -0.075, 1.0, 0.03])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.set_ticks([-vlim, -vlim / 2, 0, vlim / 2, vlim])
        cb.ax.set_xticklabels([f"{t:+.2g}" if t else "0" for t in (-vlim, -vlim / 2, 0, vlim / 2, vlim)])
        cb.ax.tick_params(labelsize=6.6, length=0, colors=H.INK2)
        cb.set_label(cbar_label, fontsize=7, color=H.INK2)
    H.frame(fig, number, section, title, takeaway, source, top=0.80, bottom=0.12, left=0.07, right=0.98)
    fig.subplots_adjust(wspace=0.12)
    return fig


def drawdown_chart(paths: dict, *, number: str, title: str, takeaway: str, colours: dict,
                   section: str = "Out of sample", source: str = SOURCE_EST):
    """Underwater chart: each daily value path as a fall from its running peak, trough marked and labelled."""
    import matplotlib.dates as mdates
    fig, ax = H.figure(4.2)
    troughs = []
    for name, v in paths.items():
        dd = v / v.cummax() - 1
        focal = colours[name] == H.GREEN
        ax.plot(dd.index, dd.values * 100, color=colours[name], lw=2.0 if focal else 1.4, label=name,
                zorder=3 if focal else 2)
        t = dd.idxmin()
        troughs.append((name, t, dd.min()))
        ax.plot([t], [dd.min() * 100], "o", ms=5, color=colours[name], mec=H.PAPER, mew=1.2, zorder=4)
    lo = min(d for _, _, d in troughs) * 100
    # trough labels to the right of each dot, pushed apart vertically so troughs on the same day never collide
    gap = abs(lo) * 0.075
    placed = []
    for name, t, d in sorted(troughs, key=lambda x: x[2]):
        y = d * 100
        for py in placed:
            if abs(y - py) < gap:
                y = py + gap
        placed.append(y)
        ax.annotate(f"{name} {d:.0%}", (t, d * 100), xytext=(t + pd.Timedelta(days=10), y), textcoords="data",
                    fontsize=7.2, color=colours[name], fontweight="semibold", va="center",
                    arrowprops=dict(arrowstyle="-", color=colours[name], lw=0.6, shrinkA=0, shrinkB=3)
                    if abs(y - d * 100) > 0.5 else None)
    ax.axhline(0, color=H.INK3, lw=0.7)
    ax.set_ylim(lo * 1.18, 3)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.margins(x=0)
    H.grid(ax, "y")
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    H.frame(fig, number, section, title, takeaway, source, left=0.08, right=0.97)
    return fig


def forecast_check(df, *, colours: dict, number: str, title: str, takeaway: str,
                   section: str = "Out of sample", source: str = SOURCE_EST):
    """Two dumbbell panels: forecast against realised value, with the realised value's 95% interval.

    df: one row per portfolio, columns ex_ante_return, realised_return, ret_hw, ex_ante_vol, realised_vol,
    vol_hw (decimals). Rows without a forecast show the realised value only. A forecast outside the
    interval is marked, since that is the one comparison a short sample can decide.
    """
    from matplotlib.lines import Line2D
    fig, axes = H.figure(4.6, ncols=2, sharey=True)
    names = list(df.index)
    y = np.arange(len(names))[::-1]
    panels = [("Annual return", "ex_ante_return", "realised_return", "ret_hw"),
              ("Annual volatility", "ex_ante_vol", "realised_vol", "vol_hw")]
    for ax, (sub, fc, rc, hc) in zip(axes, panels):
        for yi, name in zip(y, names):
            c, r, hw, f = colours[name], df.loc[name, rc] * 100, df.loc[name, hc] * 100, df.loc[name, fc]
            ax.plot([r - hw, r + hw], [yi, yi], color=c, lw=6, alpha=0.22, solid_capstyle="butt", zorder=1)
            if pd.notna(f):
                f *= 100
                ax.plot([f, r], [yi, yi], color=c, lw=1.2, zorder=2)
                ax.plot([f], [yi], "o", ms=8, mfc=H.PAPER, mec=c, mew=1.8, zorder=4)
                if abs(f - r) > hw:
                    ax.annotate("forecast outside interval", (r, yi), xytext=(0, -12), textcoords="offset points",
                                ha="center", va="top", fontsize=6.9, color=H.RED, fontweight="semibold")
            ax.plot([r], [yi], "o", ms=8, color=c, mec=H.PAPER, mew=1.2, zorder=5)
        ax.set_title(sub, loc="left", fontsize=8.8, fontweight="semibold", color=H.INK, pad=8)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        H.grid(ax, "x")
        ax.axvline(0, color=H.INK3, lw=0.6, zorder=0)
        ax.set_ylim(y.min() - 0.7, y.max() + 0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontweight="semibold")
    lo = min(0, (df["realised_vol"] - df["vol_hw"]).min() * 100)
    axes[1].set_xlim(lo, (df["realised_vol"] + df["vol_hw"]).max() * 100 * 1.08)
    handles = [Line2D([], [], ls="", marker="o", ms=7, mfc=H.PAPER, mec=H.INK2, mew=1.6, label="Forecast at end-2024"),
               Line2D([], [], ls="", marker="o", ms=7, color=H.INK2, label="Realised, Jan 2025 to Sep 2026"),
               Line2D([], [], color=H.INK2, lw=6, alpha=0.22, label="Realised, 95% interval")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.4, bbox_to_anchor=(0.55, 0.055))
    H.frame(fig, number, section, title, takeaway, source, top=0.77, left=0.17, right=0.97, bottom=0.19)
    fig.subplots_adjust(wspace=0.08)
    return fig


def fan_chart(years: np.ndarray, quantiles: dict, medians: dict, *, colours: dict, start_value: float,
              number: str, title: str, takeaway: str, dashes: dict | None = None,
              section: str = "Monte Carlo", source: str = SOURCE_EST):
    """Projected wealth on a log scale: the focal scenario's 5-95% and 25-75% bands and median,
    plus the medians of the other scenarios, each labelled at the right edge."""
    fig, ax = H.figure(4.6)
    q = quantiles
    ax.fill_between(years, q[0.05], q[0.95], color=H.GREEN, alpha=0.10, lw=0, zorder=1)
    ax.fill_between(years, q[0.25], q[0.75], color=H.GREEN, alpha=0.20, lw=0, zorder=1)
    ends = {}
    for name, med in medians.items():
        focal = colours[name] == H.GREEN
        ax.plot(years, med, color=colours[name], lw=2.2 if focal else 1.5, ls=(dashes or {}).get(name, "-"),
                zorder=3 if focal else 2)
        ends[name] = med[-1]
    ax.axhline(start_value, color=H.INK3, lw=0.7, ls=(0, (2, 3)), zorder=1)
    ax.set_yscale("log")
    lo, hi = min(q[0.05].min(), min(ends.values())) * 0.85, max(q[0.95].max(), max(ends.values())) * 1.25
    ax.set_ylim(lo, hi)
    from matplotlib.ticker import FixedLocator, NullLocator
    ticks = [v for v in (1e5, 2.5e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7) if lo <= v <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(_money_fmt())
    # end labels in log space, pushed apart
    import math
    items = sorted(((math.log10(v), n) for n, v in ends.items()))
    gap, placed = (math.log10(hi) - math.log10(lo)) * 0.075, []
    for y, n in items:
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.text(years[-1] + 0.15, 10 ** y, f"{n}\n${ends[n] / 1e3:,.0f}k" if ends[n] < 1e6 else f"{n}\n${ends[n] / 1e6:,.1f}m",
                va="center", fontsize=7.4, color=colours[n], fontweight="semibold", clip_on=False)
    k = int(len(years) * 0.55)
    ax.text(years[k], q[0.05][k] * 1.04, "5% of outcomes below this edge", fontsize=6.8, color=H.GREEN_DARK, va="bottom")
    ax.text(years[k], q[0.25][k] * 1.03, "middle half of outcomes", fontsize=6.8, color=H.GREEN_DARK, va="bottom")
    ax.set_xlim(0, years[-1])
    ax.set_xlabel("Years from 30 Sep 2026")
    H.grid(ax, "y")
    ax.spines["left"].set_visible(False)
    H.frame(fig, number, section, title, takeaway, source, left=0.09, right=0.78)
    return fig
