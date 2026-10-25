"""
eval/plots.py — Shared figure style, so every figure in the report looks the same.

LMS is always blue and NLMS always orange; a sweep of one algorithm uses shades of
its colour (light = small parameter value, dark = large).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

COLOR_LMS = "#2a78d6"
COLOR_NLMS = "#eb6834"
COLOR_SIGNAL = "#8a8a85"
COLOR_NEAR_ONLY = "#efeee9"
COLOR_DT = "#dcdad2"
COLOR_THEORY = "#3d3d3a"

DPI = 150


def shades(base, n):
    """n shades of a hex colour, from light to the colour itself."""
    rgb = np.array(matplotlib.colors.to_rgb(base))
    return [tuple(1 - (1 - rgb) * f) for f in np.linspace(0.45, 1.0, n)]


def style(ax, title=None, xlabel=None, ylabel=None):
    """Grid, no top/right frame, left-aligned title."""
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, path):
    """tight_layout, write at report quality, close, and say where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {path}")
