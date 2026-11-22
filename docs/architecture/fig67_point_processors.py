#!/usr/bin/env python
"""Figures 6 and 7: the point-processor block, transformer and MLP variants.

Identical geometry; only the mixing sublayer differs. Checked against
projects/proposal_first_pidray_occ/models/point_refine/transformer.py (_EncoderBlock,
nn.MultiheadAttention: 4 heads x 32, Q/K/V in-projection, output projection;
attention within one RoI, the RoI axis being the batch axis) and mlp_control.py
(_MixerBlock: Linear 128->256, GELU, Linear 256->128, point-wise). Both blocks are
pre-LN with a residual around each sublayer; FFN is Linear 128->512, GELU,
Linear 512->128; dropout 0; two blocks with separate parameters.
"""
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

from draw_common import Canvas, DATA, OP, FIXED, SOFT, legend

HERE = Path(__file__).resolve().parent
LBL, ANN = 5.6, 4.8
YM, BH = 17.4, 4.2            # main flow line, standard box height


def draw(kind: str, stem: str) -> None:
    c = Canvas(100, 34, 7.0)
    title = "Transformer block" if kind == "transformer" else "MLP block"

    # ---------------------------------------------------- input / output tokens
    for x, name in ((0.4, "$Z$"), (92.0, "$Z'$")):
        c.box(x, YM - BH / 2, 4.6, BH, name, DATA, fs=LBL + 0.4)
        c.label(x + 2.3, YM - BH / 2 - 1.9, "$P$ points\n128 ch", fs=ANN)

    # ---------------------------------------------------- block frame
    c.container(6.4, 6.6, 83.6, 21.8, title)
    c.label(89.2, 27.35, "× 2  ·  sequential  ·  separate parameters", fs=ANN,
            ha="right", style="italic", color=SOFT)

    # ---------------------------------------------------- residual skeleton
    J1, J2, A1, A2, RY = 7.1, 52.9, 50.9, 88.0, 8.4
    c.route([(5.0, YM), (9.25, YM)])
    c.dot(J1, YM, r=0.36)
    c.box(9.4, YM - BH / 2, 6.5, BH, "LayerNorm", OP, fs=5.2)
    c.oplus(A1, YM, r=1.1)
    c.route([(A1 + 1.1, YM), (54.85, YM)])
    c.dot(J2, YM, r=0.36)
    c.box(55.0, YM - BH / 2, 6.5, BH, "LayerNorm", OP, fs=5.2)
    c.oplus(A2, YM, r=1.1)
    c.route([(A2 + 1.1, YM), (91.85, YM)])
    for j, a in ((J1, A1), (J2, A2)):
        c.route([(j, YM), (j, RY), (a, RY), (a, YM - 1.25)])

    # ---------------------------------------------------- mixing sublayer
    MX0, MX1, MY0, MY1 = 16.4, 49.2, 10.2, 26.2
    if kind == "transformer":
        c.container(MX0, MY0, MX1 - MX0, MY1 - MY0)
        c.label(MX0 + 1.2, MY1 - 1.2, "Multi-head self-attention", fs=LBL, ha="left", weight="bold")
        c.label(MX0 + 1.2, MY1 - 2.45, "among the $P$ points of one RoI", fs=ANN, ha="left",
                style="italic", color=SOFT)
        c.route([(15.9, YM), (17.6, YM)], head=False)
        c.dot(17.6, YM, r=0.36)
        ys = {"Q": YM + 3.1, "K": YM, "V": YM - 3.1}
        c.route([(17.6, ys["Q"]), (17.6, ys["V"])], head=False)
        for k, y in ys.items():
            c.route([(17.6, y), (18.85, y)])
            c.box(19.0, y - 1.3, 4.6, 2.6, f"$W_{k}$", OP, fs=LBL, rounding=0.45)
            c.route([(23.6, y), (26.35, y)])
            c.label(25.0, y + 0.75, f"${k}$", fs=4.6)
        # four heads: stacked sheets behind the front one
        AX, AY, AW, AH = 26.5, YM - 4.2, 8.5, 8.4
        for i in (3, 2, 1):
            d = 0.35 * i
            c.ax.add_patch(FancyBboxPatch((AX + d + 0.5, AY + d + 0.5), AW - 1.0, AH - 1.0,
                                          boxstyle="round,pad=0.5,rounding_size=0.5",
                                          fc=FIXED, ec="black", lw=0.55, zorder=2))
        c.box(AX, AY, AW, AH, "Attention", FIXED, fs=LBL, rounding=0.5)
        c.label(AX + AW / 2, AY - 1.3, "4 heads · 32 ch per head", fs=ANN)
        c.route([(AX + AW + 1.05, YM), (37.85, YM)])
        c.box(38.0, YM - BH / 2, 4.6, BH, "Concat", FIXED, fs=5.2)
        c.route([(42.6, YM), (44.05, YM)])
        c.box(44.2, YM - BH / 2, 4.2, BH, "$W_O$", OP, fs=LBL)
        c.route([(48.4, YM), (A1 - 1.25, YM)])
    else:
        c.container(MX0, MY0, MX1 - MX0, MY1 - MY0)
        c.label(MX0 + 1.2, MY1 - 1.2, "Point-wise MLP", fs=LBL, ha="left", weight="bold")
        c.label(MX0 + 1.2, MY1 - 2.45, "each of the $P$ points independently", fs=ANN, ha="left",
                style="italic", color=SOFT)
        c.route([(15.9, YM), (19.35, YM)])
        c.box(19.5, YM - BH / 2, 6.5, BH, "Linear", OP, fs=LBL)
        c.route([(26.0, YM), (29.35, YM)])
        c.box(29.5, YM - BH / 2, 5.0, BH, "GELU", FIXED, fs=5.2)
        c.label(32.0, YM - BH / 2 - 1.5, "hidden 256 ch", fs=ANN)
        c.route([(34.5, YM), (37.85, YM)])
        c.box(38.0, YM - BH / 2, 6.5, BH, "Linear", OP, fs=LBL)
        c.route([(44.5, YM), (A1 - 1.25, YM)])

    # ---------------------------------------------------- feed-forward sublayer
    c.container(62.4, MY0, 23.8, MY1 - MY0)
    c.label(63.6, MY1 - 1.2, "Feed-forward network", fs=LBL, ha="left", weight="bold")
    c.label(63.6, MY1 - 2.45, "each point independently", fs=ANN, ha="left", style="italic", color=SOFT)
    c.route([(61.5, YM), (63.85, YM)])
    c.box(64.0, YM - BH / 2, 6.0, BH, "Linear", OP, fs=LBL)
    c.route([(70.0, YM), (72.15, YM)])
    c.box(72.3, YM - BH / 2, 4.7, BH, "GELU", FIXED, fs=5.2)
    c.label(74.65, YM - BH / 2 - 1.5, "hidden 512 ch", fs=ANN)
    c.route([(77.0, YM), (79.15, YM)])
    c.box(79.3, YM - BH / 2, 6.0, BH, "Linear", OP, fs=LBL)
    c.route([(85.3, YM), (A2 - 1.25, YM)])

    # ---------------------------------------------------- legend and frame
    Y0, Y1 = 2.6, 31.0
    c.ax.set_ylim(Y0, Y1)
    c.fig.set_size_inches(7.0, (Y1 - Y0) * 7.0 / 100)
    legend(c, 0.8, 4.3, [(OP, "learned module"), (FIXED, "parameter-free operation"),
                         (DATA, "tensor")])
    c.save(HERE, stem)


draw("transformer", "fig6_transformer_point_processor")
draw("mlp", "fig7_mlp_point_processor")
