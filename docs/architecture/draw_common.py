#!/usr/bin/env python
"""Shared drawing primitives for the three architecture figures.

Deterministic vector graphics: every coordinate is a literal in the figure
scripts, nothing is laid out by an automatic graph engine, so re-running the
scripts reproduces byte-identical geometry. The drawing units are square: a
figure declares ``W`` units of width and ``FW`` inches, and the height follows,
so circles are round and a unit means the same in x and y.

Palette follows the manuscript convention: pale blue = data/tensors,
pale peach = learned operations (two shades, the deeper one for the point
mixer and the feed-forward sublayer), white = containers.
"""
from __future__ import annotations

from pathlib import Path as _Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,          # embed TrueType, keeps text selectable in the PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",      # keep <text> elements, so the SVG stays editable
    "svg.hashsalt": "xrayseg-architecture",
    "path.simplify": False,
})

DATA = "#e6eef8"    # pale blue: tensors and data
OP = "#fbe7cf"      # pale peach: learned operations
OP2 = "#f7d3ab"     # deeper peach: point mixer and feed-forward sublayers
NEU = "#ffffff"     # containers
EDGE = "#000000"
SOFT = "#5a5a5a"    # residual / auxiliary connectors


class Canvas:
    """A figure with square drawing units."""

    def __init__(self, width_units: float, height_units: float, width_inches: float):
        self.W, self.H = float(width_units), float(height_units)
        scale = width_inches / self.W
        self.fig, self.ax = plt.subplots(figsize=(width_inches, self.H * scale))
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.patch.set_facecolor("white")

    # -- primitives ------------------------------------------------------
    def box(self, x, y, w, h, text="", fc=NEU, fs=6.2, lw=0.8, ls="-", weight="normal",
            ec=EDGE, rounding=0.7, pad=0.0, va="center", ty=None):
        self.ax.add_patch(FancyBboxPatch(
            (x + rounding, y + rounding), w - 2 * rounding, h - 2 * rounding,
            boxstyle=f"round,pad={rounding},rounding_size={rounding}",
            fc=fc, ec=ec, lw=lw, ls=ls, mutation_aspect=1.0, zorder=2))
        if text:
            self.ax.text(x + w / 2, y + h / 2 if ty is None else ty, text, ha="center",
                         va=va, fontsize=fs, weight=weight, linespacing=1.25, zorder=4)
        return (x, y, w, h)

    def container(self, x, y, w, h, title="", fs=6.9, lw=0.7, ls="-", ec="#6f6f6f"):
        self.ax.add_patch(FancyBboxPatch(
            (x + 0.8, y + 0.8), w - 1.6, h - 1.6,
            boxstyle="round,pad=0.8,rounding_size=0.8",
            fc=NEU, ec=ec, lw=lw, ls=ls, mutation_aspect=1.0, zorder=1))
        if title:
            self.ax.text(x + w / 2, y + h + 0.9, title, ha="center", va="bottom",
                         fontsize=fs, weight="bold", zorder=4)
        return (x, y, w, h)

    def route(self, pts, lw=0.75, ms=5.5, color=EDGE, ls="-", head=True, zorder=3):
        """Orthogonal (or any) polyline, arrowhead on the final segment."""
        verts = [(float(a), float(b)) for a, b in pts]
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        self.ax.add_patch(FancyArrowPatch(
            path=Path(verts, codes), arrowstyle="-|>" if head else "-",
            mutation_scale=ms, lw=lw, color=color, linestyle=ls,
            shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt",
            zorder=zorder, fill=True))

    def dot(self, x, y, r=0.42, color=EDGE):
        self.ax.add_patch(Circle((x, y), r, fc=color, ec=color, lw=0, zorder=5))

    def oplus(self, x, y, r=1.05, fs=7.0):
        self.ax.add_patch(Circle((x, y), r, fc="white", ec=EDGE, lw=0.8, zorder=4))
        self.ax.text(x, y, "+", ha="center", va="center", fontsize=fs, zorder=5)
        return (x, y, r)

    def label(self, x, y, text, fs=5.7, ha="center", va="center", style="normal",
              weight="normal", color=EDGE, rotation=0):
        self.ax.text(x, y, text, ha=ha, va=va, fontsize=fs, style=style, weight=weight,
                     color=color, rotation=rotation, linespacing=1.2, zorder=6)

    def grid_patch(self, cx, cy, n=6, cell=1.25, hot=(), on="#d0342c", off="#e9e9e9"):
        x0, y0 = cx - n * cell / 2, cy - n * cell / 2
        for i in range(n):
            for j in range(n):
                self.ax.add_patch(Rectangle(
                    (x0 + j * cell, y0 + i * cell), cell, cell,
                    fc=on if (i, j) in hot else off, ec="#8c8c8c", lw=0.4, zorder=3))
        return (x0, y0, n * cell, n * cell)

    # -- output ----------------------------------------------------------
    def save(self, out_dir, stem, dpi=600):
        out = _Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        for ext, meta in (("svg", {"Date": None}), ("pdf", {"CreationDate": None}),
                          ("png", None)):
            kw = {"metadata": meta} if meta else {}
            self.fig.savefig(out / f"{stem}.{ext}", dpi=dpi, facecolor="white", **kw)
        plt.close(self.fig)
        print(f"wrote {stem}.svg/.pdf/.png")


# ---------------------------------------------------------------- additions for figs 5-7
FIXED = "#ffffff"   # parameter-free operation (no learned weights)


def legend(c: Canvas, x: float, y: float, items, fs=5.0, gap=3.0):
    """One row of colour swatches: [(fill, text), ...]. Returns the end x."""
    c.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)   # measure in the final frame
    for fill, text in items:
        c.box(x, y - 0.75, 2.6, 1.5, "", fill, lw=0.6, rounding=0.3)
        t = c.ax.text(x + 3.3, y, text, ha="left", va="center", fontsize=fs, zorder=6)
        c.fig.canvas.draw()
        bb = t.get_window_extent().transformed(c.ax.transData.inverted())
        x = bb.x1 + gap
    return x


def hop_route(c: Canvas, pts, hop_x, hop_y, r=0.7, **kw):
    """Orthogonal route with a semicircular hop where it crosses another line.

    The hop is taken on the first horizontal segment that spans hop_x at hop_y.
    """
    import numpy as _np
    from matplotlib.path import Path as _Path
    verts, codes = [pts[0]], [_Path.MOVETO]
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if ya == yb == hop_y and min(xa, xb) < hop_x < max(xa, xb):
            sgn = 1 if xb > xa else -1
            verts.append((hop_x - sgn * r, ya)); codes.append(_Path.LINETO)
            for t in _np.linspace(_np.pi, 0, 13)[1:] if sgn > 0 else _np.linspace(0, _np.pi, 13)[1:]:
                verts.append((hop_x + r * _np.cos(t), ya + r * _np.sin(t))); codes.append(_Path.LINETO)
        verts.append((xb, yb)); codes.append(_Path.LINETO)
    c.ax.add_patch(FancyArrowPatch(
        path=_Path(verts, codes), arrowstyle="-|>", mutation_scale=kw.get("ms", 5.5),
        lw=kw.get("lw", 0.75), color=kw.get("color", EDGE), shrinkA=0, shrinkB=0,
        joinstyle="miter", capstyle="butt", zorder=3))
