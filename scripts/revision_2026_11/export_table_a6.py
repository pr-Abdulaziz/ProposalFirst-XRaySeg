#!/usr/bin/env python
"""Export Table A6 (inference-time ablations incl. the 196/392/784/1568 point sweep) from the verified
ablation/summary.json as Markdown and LaTeX. Numbers only from that file; nothing recomputed."""
import json
from pathlib import Path
REV = Path(__file__).resolve().parents[2] / "output/pixray_multiseed_bs4_e50_2026-09-14/revision_2026-11"
S = json.loads((REV / "ablation/summary.json").read_text())
A, C = S["arms"], S["contrasts"]
ROWS = [(None, "Full refiner (published: 784 points, 2 levels, all stages)"), ("off", "Refinement disabled"),
        ("random784", "Random instead of uncertain points (784)"), ("laststage", "Refine final cascade stage only"),
        ("level56", "One refinement level (28→56)"), ("pts196", "196 points (training budget)"),
        ("pts392", "392 points"), ("pts1568", "1,568 points")]
f = lambda x: f"{x:+.2f}".replace("-", "−")
ci = lambda c: f"[{f(c[0])}, {f(c[1])}]"
md = ["| Variant | Transformer AP | Δ vs full | Δ vs Parent | MLP AP | Δ vs full | Δ vs Parent |", "|---|---|---|---|---|---|---|"]
tex = [r"\begin{tabular}{lcccccc}\toprule",
       r" & \multicolumn{3}{c}{Transformer refiner} & \multicolumn{3}{c}{MLP refiner}\\",
       r"Variant & AP & $\Delta$ vs full [95\% CI] & $\Delta$ vs Parent [95\% CI] & AP & $\Delta$ vs full [95\% CI] & $\Delta$ vs Parent [95\% CI]\\\midrule"]
for v, label in ROWS:
    cells_md, cells_tex = [], []
    for m in ("09", "09M"):
        arm = m if v is None else f"{m}:{v}"
        ap = A[arm]["AP"]["mean"]
        vp = C[arm]["vs_parent"]["AP"]
        vf = C[arm].get("vs_full", {}).get("AP")
        full = "—" if vf is None else f"{f(vf['delta'])} {ci(vf['ci95'])}"
        par = f"{f(vp['delta'])} {ci(vp['ci95'])}"
        cells_md += [f"{ap:.2f}", full, par]
        cells_tex += [f"{ap:.2f}", full.replace("−", "$-$"), par.replace("−", "$-$")]
    md.append(f"| {label} | " + " | ".join(cells_md) + " |")
    tex.append(label.replace("→", r"$\to$").replace("1,568", "1{,}568") + " & " + " & ".join(cells_tex) + r"\\")
tex += [r"\bottomrule\end{tabular}"]
note = (f"Mask AP (%), mean of three seeds; Parent = {A['03']['AP']['mean']:.2f}. Δ: mean of per-seed paired differences; "
        "95% CI: two-level paired bootstrap (seeds, then images; B = 2,000). Post-hoc, inference-time analyses of the "
        "trained checkpoints on the official test split; no model was retrained and no setting was selected from these results.")
(REV / "table_A6.md").write_text("# Table A6. Inference-time ablations of the trained refiners\n\n" + "\n".join(md) + "\n\n" + note + "\n")
(REV / "table_A6.tex").write_text("% Table A6 (from ablation/summary.json)\n" + "\n".join(tex) + "\n% " + note + "\n")
print("\n".join(md)); print(note)
