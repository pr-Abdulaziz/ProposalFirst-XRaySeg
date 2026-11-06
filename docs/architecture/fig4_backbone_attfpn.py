#!/usr/bin/env python
"""Figure 4: ResNet-101 + AttFPN feature hierarchy (overview).

Module names inside blocks; feature properties outside, every number labelled.
Checked against the resolved config of method 03 and
projects/proposal_first_pidray_occ/models/attfpn.py:
  ResNet(depth=101, frozen_stages=1, norm_eval=True, out_indices=(0,1,2,3))
  AttFPN: FPN (256/512/1024/2048 -> 256, 5 levels, P6 = max-pool of P5),
          then one selective-attention module per output level
  RPN: strides 4-64; RoIAlign: strides 4-32 (box 7x7, mask 14x14); point refiner: P2
"""
from pathlib import Path

from draw_common import Canvas, DATA, OP, SOFT

HERE = Path(__file__).resolve().parent
FROZEN = "#eeeeee"
LBL = 6.0      # block and tensor names
ANN = 5.0      # supporting annotations
c = Canvas(100, 36, 7.0)

c.container(8.8, 3.0, 46.7, 29.0, "ResNet-101 backbone")
c.container(57.0, 3.0, 25.0, 29.0, "AttFPN neck")
c.container(83.5, 3.0, 16.2, 29.0, "Output pyramid")

# ---------------------------------------------------------------- input and stages
SY, SH = 22.8, 5.4
MID = SY + SH / 2
# Input: a real PIXray training image containing a prohibited item (a pistol),
# train/102036201.png, cropped to the bag. Stored beside the figure so the
# drawing does not depend on the dataset path.
from matplotlib.image import imread
from matplotlib.patches import Rectangle
thumb = imread(str(HERE / "assets" / "fig4_input_pixray_train_102036201.png"))
IW = 10.4
IH = IW * thumb.shape[0] / thumb.shape[1]
IX, IY = -2.2, MID - IH / 2
c.ax.imshow(thumb, extent=(IX, IX + IW, IY, IY + IH), zorder=2, interpolation="lanczos")
c.ax.add_patch(Rectangle((IX, IY), IW, IH, fill=False, ec="#8c8c8c", lw=0.5, zorder=3))
c.label(IX + IW / 2, IY - 1.3, "$H \\times W \\times 3$", fs=ANN)
c.route([(IX + IW, MID), (10.05, MID)])

stages = [(10.2, 6.0, "Stem", True), (18.2, 5.6, "Res2", True), (27.2, 5.6, "Res3", False),
          (36.2, 5.6, "Res4", False), (45.2, 5.6, "Res5", False)]
for x, w, text, frozen in stages:
    c.box(x, SY, w, SH, text, FROZEN if frozen else OP, fs=LBL)
for (x1, w1, *_), (x2, *_) in zip(stages, stages[1:]):
    c.route([(x1 + w1, MID), (x2 - 0.15, MID)])

bx0, bx1, by = 10.2, 23.8, SY + SH + 1.0
c.route([(bx0, by - 0.5), (bx0, by), (bx1, by), (bx1, by - 0.5)], head=False, color=SOFT, lw=0.6)
c.label((bx0 + bx1) / 2, by + 1.1, "frozen", fs=ANN, style="italic", color=SOFT)
c.label(54.8, by + 1.1, "BatchNorm statistics frozen in all stages", fs=ANN,
        ha="right", style="italic", color=SOFT)

# ---------------------------------------------------------------- C2-C5
feats = [("C_2", "256 ch", "stride 4"), ("C_3", "512 ch", "stride 8"),
         ("C_4", "1024 ch", "stride 16"), ("C_5", "2048 ch", "stride 32")]
CW, CH, CY = 4.4, 3.0, 16.2
CMID = CY + CH / 2
bus_y = [6.3, 7.8, 9.3, 10.8]        # C2 lowest ... C5 highest: no crossings
FPN_X = 60.0
for (x, w, *_), (name, ch, stride), yb in zip(stages[1:], feats, bus_y):
    cx = x + w / 2
    c.route([(cx, SY), (cx, CY + CH + 0.15)])
    c.box(cx - CW / 2, CY, CW, CH, f"${name}$", DATA, fs=LBL, rounding=0.5)
    c.label(cx, CY - 2.1, f"{ch}\n{stride}", fs=ANN)
    # leave from the side so the annotation beneath stays clear of the line
    xr = cx + CW / 2 + 1.1
    c.route([(cx + CW / 2, CMID), (xr, CMID), (xr, yb), (FPN_X - 0.15, yb)])

# ---------------------------------------------------------------- AttFPN
NW = 13.0
c.box(FPN_X, 5.3, NW, 6.3, "FPN fusion", OP, fs=LBL)

P_Y = [29.2, 26.4, 23.6, 20.8, 18.0]
SA_MID = P_Y[2]
c.box(FPN_X, SA_MID - 2.7, NW, 5.4, "Selective attention", OP, fs=LBL)
c.route([(FPN_X + NW / 2, 11.6), (FPN_X + NW / 2, SA_MID - 2.85)])
c.label(FPN_X + NW / 2 + 0.8, 16.3, "5 levels", fs=ANN, ha="left")

# ---------------------------------------------------------------- outputs
BUS, PX, PW, PH = 80.3, 85.0, 4.2, 2.3
c.route([(FPN_X + NW, SA_MID), (BUS, SA_MID)], head=False)
c.route([(BUS, P_Y[0]), (BUS, P_Y[-1])], head=False)
for i, y in enumerate(P_Y):
    c.route([(BUS, y), (PX - 0.15, y)])
    if 0 < i < len(P_Y) - 1:
        c.dot(BUS, y, r=0.35)
    c.box(PX, y - PH / 2, PW, PH, f"$P_{i + 2}$", DATA, fs=LBL, rounding=0.45)
    c.label(PX + PW + 0.7, y, f"stride {4 * 2 ** i}", fs=ANN, ha="left")
c.label(91.6, 15.9, "256 channels per level", fs=ANN, style="italic", color=SOFT)

c.ax.plot([PX - 0.5, 98.7], [14.5, 14.5], color="#c4c4c4", lw=0.5, zorder=2)
for y, t in ((12.8, "RPN: $P_2$–$P_6$"), (10.9, "RoIAlign: $P_2$–$P_5$"),
             (9.0, "Point refiner: $P_2$")):
    c.label(91.6, y, t, fs=ANN)
c.label(91.6, 5.8, "Box RoI: 7 × 7\nmask RoI: 14 × 14", fs=4.6, style="italic", color=SOFT)

# trim the empty margins left after the footer was removed
# the canvas extends slightly left of 0 to give the input image room
X0, X1, Y0, Y1 = -2.6, 100.0, 2.0, 35.0
c.ax.set_xlim(X0, X1)
c.ax.set_ylim(Y0, Y1)
c.fig.set_size_inches(7.0, (Y1 - Y0) * 7.0 / (X1 - X0))
c.save(HERE, "fig4_backbone_attfpn")
