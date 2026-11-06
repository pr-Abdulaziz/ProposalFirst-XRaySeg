#!/usr/bin/env python
"""Figure 1: overall framework, with an expanded view of one refinement level.

Every label is checked against the implementation in
``projects/proposal_first_pidray_occ/models/point_refine/``:
head.py (refinement loop), embeddings.py (token construction),
sampling.py (selection and scatter).
"""
from pathlib import Path

from draw_common import Canvas, DATA, OP, SOFT

HERE = Path(__file__).resolve().parent
c = Canvas(100, 62, 7.0)

# ---------------------------------------------------------------- top row
TY, TH = 53.9, 6.6
top = [
    (0.8, 7.0, "Image", DATA, "normal", 6.1),
    (9.8, 11.5, "ResNet-101", OP, "normal", 6.1),
    (23.3, 12.0, "AttFPN\n$P_2$–$P_6$", OP, "normal", 6.1),
    (37.3, 13.5, "RPN + 3-stage\ncascade, $s{=}1,2,3$", OP, "normal", 6.1),
    (52.8, 13.8, "3 separate coarse\nmask heads, $M^{(s)}$\n$15{\\times}28{\\times}28$", OP, "normal", 5.7),
    (70.0, 12.6, "Shared point\nrefiner", OP, "bold", 6.1),
    (84.6, 14.6, "Refined masks\n$M^{(s,112)}$\n$15{\\times}112{\\times}112$", DATA, "normal", 5.7),
]
# stacked outline behind the mask-head box: three separate heads, one per stage
for k in (2, 1):
    c.ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
        (52.8 + 0.7 + 0.6 * k, TY + 0.7 + 0.6 * k), 13.8 - 1.4, TH - 1.4,
        boxstyle="round,pad=0.7,rounding_size=0.7", fc=OP, ec="black", lw=0.6, zorder=1))
for x, w, t, fc, weight, fs in top:
    c.box(x, TY, w, TH, t, fc, fs=fs, weight=weight)
for (x1, w1, *_), (x2, *_) in zip(top, top[1:]):
    start = x1 + w1 + (1.3 if x1 == 52.8 else 0.0)   # clear the stacked mask heads
    c.route([(start, TY + TH / 2), (x2 - 0.15, TY + TH / 2)])

# ------------------------------------------------- expanded refinement view
c.ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
    (1.3, 3.0), 97.4, 47.5, boxstyle="round,pad=0.8,rounding_size=0.8",
    fc="white", ec="#6f6f6f", lw=0.7, ls=(0, (4, 2.4)), zorder=0))
c.label(2.6, 49.4, "Refinement at level $r\\in\\{56,112\\}$", fs=6.5, ha="left",
        va="top", weight="bold")

# feeds from the detector into the expanded view
c.route([(29.3, TY), (29.3, 52.1), (50.0, 52.1), (50.0, 48.15)])       # P2
c.route([(59.7, TY), (59.7, 52.9), (63.0, 52.9), (63.0, 48.15)])     # original M^(s)
c.label(48.8, 50.4, "$P_2$", fs=6.0, ha="right")
c.label(64.2, 50.4, "$M^{(s)}$ (original)", fs=6.0, ha="left")

# ---- left column: the evolving mask ------------------------------------
LX, LW = 7.5, 16.0
c.box(LX, 42.5, LW, 6.0, "coarse logits\n$M^{(s)}$, $15{\\times}28^2$", DATA)
c.box(LX, 34.0, LW, 6.0, "Bilinear\nupsample", OP)
c.box(LX, 25.5, LW, 6.0, "$\\widetilde{M}^{(s,r)}$\n$15{\\times}r{\\times}r$", DATA)
c.box(LX, 17.0, LW, 6.0, "Uncertainty\nsampling", OP)
for y0, y1 in ((42.5, 40.15), (34.0, 31.65), (25.5, 23.15)):
    c.route([(LX + LW / 2, y0), (LX + LW / 2, y1)])

# ---- right column: the point path --------------------------------------
CX = 56.5
c.box(43.5, 42.0, 26.0, 6.0, "Sample $P_2$ and $M^{(s)}$ at $u_i$", OP)
c.box(43.5, 31.5, 26.0, 6.5, "Point embedding\n$\\phi_f{+}\\phi_m{+}\\phi_p{+}e_r$, LayerNorm", OP)
c.box(74.0, 32.5, 12.0, 4.5, "$e_r$", DATA)
c.route([(74.0, 34.75), (69.65, 34.75)])
c.box(43.5, 21.5, 11.5, 6.0, "Transformer\n$\\times2$ blocks", OP, ls=(0, (3.5, 2)))
c.box(58.0, 21.5, 11.5, 6.0, "MLP\n$\\times2$ blocks", OP, ls=(0, (3.5, 2)))
c.label(CX, 24.5, "or", fs=6.2, style="italic")
c.label(CX, 19.9, "alternatives", fs=5.6, style="italic")
c.box(44.5, 12.6, 24.0, 6.0, "Decoder: LN + Linear\n$128\\rightarrow15$", OP)
c.box(34.0, 4.5, 26.0, 7.0, "Scatter\nreplace selected cells", OP)
c.box(65.0, 4.5, 17.0, 7.0, "$M^{(s,r)}$\n$15{\\times}r{\\times}r$", DATA)

# sampled inputs into the embedding
c.route([(50.0, 42.0), (50.0, 38.15)])
c.route([(63.0, 42.0), (63.0, 38.15)])
c.label(49.0, 40.0, "$f_i$·256", fs=5.7, ha="right")
c.label(64.0, 40.0, "$m_i$·15", fs=5.7, ha="left")

# tokens into the two alternative mixers, and their outputs into the decoder
c.route([(CX, 31.5), (CX, 29.5)], head=False)
c.route([(CX, 29.5), (49.25, 29.5), (49.25, 27.65)])
c.route([(CX, 29.5), (63.75, 29.5), (63.75, 27.65)])
c.label(57.6, 30.4, "$Z\\in\\mathbb{R}^{P\\times128}$", fs=5.7, ha="left")
c.route([(49.25, 21.5), (49.25, 18.75)])
c.route([(63.75, 21.5), (63.75, 18.75)])

# decoded point logits into scatter, and the scatter output
c.route([(CX, 12.6), (CX, 11.65)])
c.label(57.6, 12.0, "$\\ell$", fs=6.2, ha="left")
c.route([(60.0, 8.0), (64.85, 8.0)])

# selection outputs: coordinates up to sampling and to the coordinate branch,
# indices down to scatter. Coordinates never reach scatter.
c.route([(LX + LW, 20.0), (35.0, 20.0), (35.0, 45.0), (43.35, 45.0)])
c.route([(35.0, 34.75), (43.35, 34.75)])
c.dot(35.0, 34.75)
c.label(29.2, 21.1, "$u_i$", fs=6.0)
c.route([(LX + LW / 2, 17.0), (LX + LW / 2, 9.8), (33.85, 9.8)])
c.label(16.9, 13.6, "$S$", fs=6.0, ha="left")

# the upsampled grid also goes to scatter, as the surface being updated
c.route([(LX, 28.5), (4.8, 28.5), (4.8, 7.0), (33.85, 7.0)])
c.label(20.0, 5.9, "$\\widetilde{M}^{(s,r)}$", fs=5.9)

# the refinement loop: the updated mask returns to the next upsampling
c.route([(73.5, 4.5), (73.5, 3.6), (2.6, 3.6), (2.6, 37.0), (7.35, 37.0)], color=SOFT)
c.label(28.0, 4.35, "$r$: $56\\rightarrow112$", fs=5.7, color=SOFT)

c.label(50.0, 1.1, "$28^2\\rightarrow56^2\\rightarrow112^2$   ·   shared across $s$ and $r$", fs=6.2)
c.save(HERE, "fig1_overall_framework")
