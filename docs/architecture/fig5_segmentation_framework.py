#!/usr/bin/env python
"""Figure 5: instance-segmentation framework at inference.

Checked against projects/proposal_first_pidray_occ/models/point_refine/cascade_adapter.py
(PointRefineCascadeRoIHead.predict_mask) and mmdet CascadeRoIHead:
  * detections: class scores averaged over the three cascade stages, boxes from
    stage 3, NMS 0.5, score >= 0.05, at most 100 per image
  * every stage's mask head runs on the SAME final boxes: RoIAlign 14x14 on P2-P5
  * each stage's coarse logits pass the ONE shared point refiner (samples P2),
    then a sigmoid; the three probability maps are averaged
  * FCNMaskHead.predict_by_feat: channel of the predicted class, pasted into the
    box, thresholded at mask_thr_binary = 0.5
Input and output are real: PIXray test image 101627037.png and the predicted
mask of the MLP variant (09M, seed 42); see make_fig5_assets.py.
"""
from pathlib import Path

from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch, Rectangle

from draw_common import Canvas, DATA, OP, FIXED, SOFT, legend, hop_route

HERE = Path(__file__).resolve().parent
LBL, ANN = 5.8, 4.9
c = Canvas(100, 44, 7.0)


def image(path, cx, cy, w):
    arr = imread(str(path))
    h = w * arr.shape[0] / arr.shape[1]
    c.ax.imshow(arr, extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2),
                zorder=2, interpolation="lanczos")
    c.ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="#8c8c8c",
                             lw=0.5, zorder=3))
    return cy - h / 2, cy + h / 2


def note(x, y, text, **kw):
    c.label(x, y, text, fs=kw.pop("fs", ANN), **kw)


# ======================================================== row 1: detector
Y1, BH = 36.0, 5.4
B1 = Y1 - BH / 2
in_bot, in_top = image(HERE / "assets/fig5_input_pixray_test_101627037.png", 6.0, Y1, 11.6)
note(6.0, in_top + 1.1, "input image", style="italic", color=SOFT)

c.box(13.2, B1, 8.2, BH, "ResNet-101", OP, fs=LBL)
c.box(23.6, B1, 7.8, BH, "AttFPN", OP, fs=LBL)
c.box(35.2, B1, 12.0, BH, "RPN + cascade", OP, fs=LBL)
note(41.2, B1 + BH + 1.2, "3 stages")
c.box(50.0, B1, 8.5, BH, "$b,\\ c$", DATA, fs=LBL)
note(54.25, B1 + BH + 1.2, "boxes, classes")
c.route([(11.8, Y1), (13.05, Y1)])
c.route([(21.4, Y1), (23.45, Y1)])
c.route([(31.4, Y1), (35.05, Y1)])
note(33.3, Y1 + 1.0, "$P_2$–$P_6$", fs=4.7)
c.route([(47.2, Y1), (49.85, Y1)])

# ======================================================== row 2: mask branch
Y2 = 18.5
B2 = Y2 - BH / 2
c.box(12.4, B2, 7.0, BH, "RoIAlign", FIXED, fs=LBL)
note(15.9, B2 - 1.4, "256 ch · 14 × 14")

# plate: one branch per cascade stage
c.ax.add_patch(FancyBboxPatch((22.0, 11.3), 45.0, 13.9, boxstyle="round,pad=0.8,rounding_size=0.8",
                              fc="white", ec="#6f6f6f", lw=0.7, ls=(0, (4, 2.4)), zorder=1))
note(21.9, 24.9, "for each stage $s = 1, 2, 3$", ha="left", style="italic", color=SOFT)

c.box(23.0, B2, 7.8, BH, "Mask head", OP, fs=LBL)
note(26.9, B2 - 1.4, "separate per stage")
c.box(33.4, B2, 5.6, BH, "$M^{(s)}$", DATA, fs=LBL)
note(36.2, B2 - 1.4, "$C$ classes\n28 × 28")
c.box(41.6, B2, 10.0, BH, "", OP, lw=1.1)
c.label(46.6, Y2 + 0.85, "Point refiner", fs=LBL, weight="bold")
c.label(46.6, Y2 - 1.05, "Transformer or MLP", fs=4.6)
note(46.6, B2 - 2.1, "shared across stages\nand refinement levels\n56 × 56 → 112 × 112")
c.box(54.2, B2, 6.0, BH, "$M^{(s,112)}$", DATA, fs=LBL)
note(57.2, B2 - 1.4, "$C$ classes\n112 × 112")
c.box(62.6, B2, 3.4, BH, "$\\sigma$", FIXED, fs=LBL + 0.6)

for x0, x1 in ((19.4, 22.85), (30.8, 33.25), (39.0, 41.45), (51.6, 54.05), (60.2, 62.45)):
    c.route([(x0, Y2), (x1, Y2)])

# ======================================================== row 2: assembly
c.box(69.6, B2, 8.2, BH, "Mean over $s$", FIXED, fs=LBL)
note(73.7, B2 - 1.4, "$C$ classes\n112 × 112")
c.box(80.2, B2, 7.8, BH, "Select class\n+ paste", FIXED, fs=5.3)
note(84.1, B2 - 1.4, "$H \\times W$")
c.box(90.0, B2, 7.6, BH, "Threshold", FIXED, fs=LBL)
note(93.8, B2 - 1.4, "$p \\geq 0.5$")
c.route([(66.0, Y2), (69.45, Y2)])
c.route([(77.8, Y2), (80.05, Y2)])
c.route([(88.0, Y2), (89.85, Y2)])

out_bot, out_top = image(HERE / "assets/fig5_output_pixray_test_101627037.png", 93.8, Y1, 12.0)
note(93.8, out_top + 1.1, "predicted mask overlay", style="italic", color=SOFT)
c.route([(93.8, B2 + BH), (93.8, out_bot - 0.15)])

# ======================================================== feature and detection routing
# P2-P5 to RoIAlign
c.route([(25.2, B1), (25.2, 30.8), (14.6, 30.8), (14.6, B2 + BH + 0.15)])
note(14.0, 27.0, "$P_2$–$P_5$", ha="right", fs=4.7)
# P2 to the point refiner
c.route([(29.6, B1), (29.6, 28.2), (44.0, 28.2), (44.0, B2 + BH + 0.15)])
note(44.6, 23.9, "$P_2$", ha="left", fs=4.7)
# detections: boxes to RoIAlign (hops over the P2 line), boxes and classes to Paste
JX, JY = 54.25, 29.6
c.route([(JX, B1), (JX, JY)], head=False)
c.dot(JX, JY, r=0.38)
hop_route(c, [(JX, JY), (17.2, JY), (17.2, B2 + BH + 0.15)], hop_x=29.6, hop_y=JY)
note(17.8, 27.0, "$b$", ha="left", fs=4.9)
c.route([(JX, JY), (84.1, JY), (84.1, B2 + BH + 0.15)])
note(84.7, 26.8, "$b,\\ c$", ha="left", fs=4.9)

# ======================================================== legend
Y0, Y1c = 3.0, 43.4
c.ax.set_ylim(Y0, Y1c)
c.fig.set_size_inches(7.0, (Y1c - Y0) * 7.0 / 100)
c.label(99.6, 5.0, "inference workflow", fs=ANN, ha="right", style="italic", color=SOFT)
legend(c, 0.8, 5.0, [(OP, "learned module"), (FIXED, "parameter-free operation"), (DATA, "tensor")])

Y0, Y1c = 3.0, 43.4
c.ax.set_ylim(Y0, Y1c)
c.fig.set_size_inches(7.0, (Y1c - Y0) * 7.0 / 100)
c.save(HERE, "fig5_segmentation_framework")
