"""
House style for every exhibit: research-note layout in green and silver.

Layout (top to bottom): a kicker ("EXHIBIT 15 | PORTFOLIO CONSTRUCTION") in
dark green, a serif title saying what the chart shows, a one-line takeaway,
a green rule, the plot, and a source line.

Type: IBM Plex Serif for titles, IBM Plex Sans for everything else. Both are
under the SIL Open Font License and ship in assets/fonts/, so the exhibits
render identically on any machine without installing anything.

Colour roles, not a rainbow:
  GREEN   the series the exhibit is about (our recommended portfolio)
  SILVER  the reference it is compared with (a benchmark or the baseline model)
  STEEL   a third series, only when a chart genuinely needs one
Green and steel blue pass the colour-vision checks as a pair; silver is a
neutral, so it is used only for the reference series and is always direct
labelled. Anything beyond three series is split into small multiples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager as fm

matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import matplotlib.transforms as mt                 # noqa: E402

FONTS = Path(__file__).resolve().parents[1] / "assets" / "fonts"
for f in FONTS.glob("*.ttf"):
    fm.fontManager.addfont(str(f))
SANS, SERIF = "IBM Plex Sans", "IBM Plex Serif"

GREEN = "#0e7c55"        # focal series
GREEN_DARK = "#0b4f39"   # kicker, rules, headings
SILVER = "#8f9a96"       # reference series
SILVER_LIGHT = "#c9cfcc"
STEEL = "#3f6fb0"        # third series
RASPBERRY = "#a8456b"    # third categorical hue (clusters); validated against STEEL and AMBER
AMBER = "#b07d2b"        # fourth series (1/N); CVD-separable from green only with direct labels
RED = "#b4412f"          # losses, used sparingly
INK, INK2, INK3 = "#15201b", "#4b5651", "#7d8784"
GRID = "#e4e8e6"
PAPER = "#ffffff"

WIDTH = 7.2              # inches: the text width of an A4 page with 2 cm margins, near enough

plt.rcParams.update({
    "text.parse_math": False,          # exhibits print dollar amounts; never read "$...$" as a formula
    "font.family": SANS, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def figure(height: float = 5.0, width: float = WIDTH, **kw):
    return plt.subplots(figsize=(width, height), dpi=200, **kw)


def frame(fig, number: str, section: str, title: str, takeaway: str, source: str,
          top: float = 0.84, bottom: float = 0.11, left: float = 0.1, right: float = 0.97) -> None:
    """Kicker, title, takeaway and rule above the plot; source line below."""
    fig.text(0.02, 0.965, f"EXHIBIT {number}  |  {section.upper()}", fontsize=7.6, color=GREEN_DARK,
             fontweight="bold", fontfamily=SANS)
    fig.text(0.02, 0.925, title, fontsize=12.5, color=INK, fontfamily=SERIF, fontweight="semibold")
    fig.text(0.02, 0.888, takeaway, fontsize=8.2, color=INK2, fontfamily=SANS)
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.872, 0.872], color=GREEN_DARK, lw=1.1))
    fig.text(0.02, 0.018, source, fontsize=6.6, color=INK3, fontfamily=SANS)
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)


def grid(ax, axis: str = "x") -> None:
    getattr(ax, f"{axis}axis").grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(axis != "x")
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def side_labels(ax, ys, texts, x: float = -0.115) -> None:
    """Small grey labels to the left of the y tick labels (e.g. each ticker's cluster)."""
    tr = mt.blended_transform_factory(ax.transAxes, ax.transData)
    for y, t in zip(ys, texts):
        ax.text(x, y, t, ha="right", va="center", fontsize=7.2, color=INK3, transform=tr)


def save(fig, path: Path) -> None:
    """PNG at 300 dpi for previews and a vector PDF for the document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
