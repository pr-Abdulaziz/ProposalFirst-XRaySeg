#!/usr/bin/env python
"""Figures 2 and 3: the two point-refiner variants, drawn with identical layouts.

Only the point-mixing sublayer differs, so the two figures share this code and
every component keeps the same position. Widths are read from the modules:
embeddings.PointTokenEmbedding (256->128, 15->64->128, 2->128->128, e_r 128),
transformer.PointRefineTransformer (4 heads x 32, FFN 128->512->128, 2 blocks,
LN + Linear 128->15) and mlp_control.PointRefineMLP (128->256->128).
"""
from pathlib import Path

from draw_common import Canvas, DATA, OP, OP2, SOFT

HERE = Path(__file__).resolve().parent

GY, GH = 3.2, 22.3          # group boxes
CY = 14.2                   # main row
ROWS = (21.4, 16.6, 11.8, 7.0)
G1, G2, G3, G4 = (1.0, 33.5), (36.5, 34.0), (72.5, 12.5), (87.0, 12.5)


def draw(title, mixer_lines, mixer_note, stem):
    c = Canvas(100, 34, 7.0)
    c.label(50.0, 31.8, title, fs=7.6, weight="bold")

    # ---------------------------------------------------------- embedding
    c.container(G1[0], GY, G1[1], GH, "Embedding")
    labels = ("$f_i$ · 256", "$m_i$ · 15", "$u_i$ · 2", "$e_r$ · 128")
    proj = ("$\\phi_f$: $256{\\rightarrow}128$",
            "$\\phi_m$: $15{\\rightarrow}64{\\rightarrow}128$, GELU",
            "$\\phi_p$: $2{\\rightarrow}128{\\rightarrow}128$, GELU")
    BUS, PX, PW = 24.2, 7.8, 14.0
    for text, y in zip(labels, ROWS):
        c.label(7.0, y, text, fs=5.8, ha="right")
    c.label(7.0, ROWS[1] - 2.0, "from $M^{(s)}$", fs=5.0, ha="right", style="italic")
    for text, y in zip(proj, ROWS):
        c.box(PX, y - 2.1, PW, 4.2, text, OP, fs=5.0)
        c.route([(PX + PW, y), (BUS, y)], head=False)
    c.route([(PX, ROWS[3]), (BUS, ROWS[3])], head=False)     # e_r goes straight in
    c.route([(BUS, ROWS[0]), (BUS, ROWS[3])], head=False)
    for y in ROWS[1:3]:
        c.dot(BUS, y)
    for y in ROWS:
        c.label(22.9, y + 0.95, "128", fs=4.8)
    c.route([(BUS, CY), (25.5, CY)])
    c.oplus(26.6, CY, r=1.05)
    c.route([(27.65, CY), (29.1, CY)])
    c.box(29.2, CY - 1.8, 3.6, 3.6, "LN", OP, fs=5.8)
    c.label(29.9, 10.2, "$Z\\in\\mathbb{R}^{P\\times128}$", fs=5.5)

    # ------------------------------------------------------ refiner block
    c.container(G2[0], GY, G2[1], GH, "Refiner block  ×2 sequential")
    c.route([(32.8, CY), (37.2, CY)])
    c.box(37.3, CY - 1.8, 3.2, 3.6, "LN", OP, fs=5.8)
    c.route([(40.5, CY), (41.3, CY)])
    c.box(41.4, CY - 2.5, 9.8, 5.0, mixer_lines, OP2, fs=5.1)
    c.route([(51.2, CY), (51.4, CY)])
    c.oplus(52.55, CY, r=1.05)
    c.route([(53.6, CY), (54.3, CY)])
    c.box(54.4, CY - 1.8, 3.2, 3.6, "LN", OP, fs=5.8)
    c.route([(57.6, CY), (58.1, CY)])
    c.box(58.2, CY - 2.5, 8.8, 5.0, "FFN\n$128{\\rightarrow}512{\\rightarrow}128$", OP2, fs=5.1)
    c.route([(67.0, CY), (67.35, CY)])
    c.oplus(68.5, CY, r=1.05)
    # residual connections: start before each LN, bypass the whole sublayer
    for x_from, x_to in ((36.6, 52.55), (54.0, 68.5)):
        c.dot(x_from, CY, r=0.38, color=SOFT)
        c.route([(x_from, CY), (x_from, 21.7), (x_to, 21.7), (x_to, CY + 1.15)], color=SOFT)
    c.label(46.3, 9.2, mixer_note, fs=5.5, style="italic")

    # ------------------------------------------------------------ decoder
    c.container(G3[0], GY, G3[1], GH, "Decoder")
    c.route([(69.55, CY), (73.9, CY)])
    c.box(74.0, CY - 1.8, 3.4, 3.6, "LN", OP, fs=5.8)
    c.route([(77.4, CY), (78.7, CY)])
    c.box(78.8, CY - 2.5, 4.8, 5.0, "Linear\n$128{\\rightarrow}15$", OP, fs=5.4)
    c.label(78.8, 9.6, "$\\ell\\in\\mathbb{R}^{P\\times15}$", fs=5.5)

    # ------------------------------------------------------------ scatter
    c.container(G4[0], GY, G4[1], GH, "Scatter")
    c.route([(83.6, CY), (88.9, CY)])
    c.label(86.2, 15.4, "$\\ell$", fs=6.0)
    gx, gy, gw, gh = c.grid_patch(93.2, CY + 0.8, n=6, cell=1.3,
                                  hot={(0, 4), (1, 1), (2, 5), (3, 3), (4, 1), (5, 4)})
    c.route([(90.6, 22.4), (90.6, gy + gh + 0.15)])
    c.route([(96.0, 22.4), (96.0, gy + gh + 0.15)])
    c.label(90.0, 23.3, "$\\widetilde{M}^{(s,r)}$", fs=5.5)
    c.label(96.0, 23.3, "$S$", fs=5.8)
    c.route([(93.2, gy), (93.2, 8.8)])
    c.label(93.2, 7.6, "$M^{(s,r)}$", fs=5.8)
    c.label(93.2, 5.1, "replace\nselected cells", fs=5.0, style="italic")

    c.label(50.0, 1.3, "$P=196$ train / $784$ test, per RoI and level", fs=6.0)
    c.save(HERE, stem)


draw("Transformer point refiner", "MHSA\n$4\\times32$", "within RoI",
     "fig2_transformer_point_refiner")
draw("MLP point refiner", "Point-wise MLP\n$128{\\rightarrow}256{\\rightarrow}128$",
     "independent points", "fig3_mlp_point_refiner")
