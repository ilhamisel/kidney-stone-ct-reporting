"""Shared figure settings: a verified palette and the matplotlib theme.

Colour roles:
  zone ramp    an ordinal single-hue ramp. Zone is an ORDERED anatomical axis
               (lower to upper), not a categorical one, so it must not be drawn
               with categorical colours.
  series 1-3   categorical slots; these three are checked for all-pairs
               distinguishability, including under the common colour-vision
               deficiencies.
  ink / grid   chart chrome stays in the background.
"""
from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# ordinal zone ramp: LOWER -> MID -> UPPER
ZONE_RAMP = {"LOWER": "#86b6ef", "MID": "#2a78d6", "UPPER": "#104281"}
# categorical slots (only the first three are all-pairs verified)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"


def apply_theme() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 600, "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
