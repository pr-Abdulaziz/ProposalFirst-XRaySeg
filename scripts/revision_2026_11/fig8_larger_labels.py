#!/usr/bin/env python
"""Fig. 8 redrawn with larger labels. Reuses make_qualitative_final.py unchanged:
same Data(), same fixed selection rules, same rows and panels; only font sizes and row height change.
Output goes to revision_2026-11/figures/. Verifies rows/panels equal the published figure's record."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import scripts.campaign_multiseed.make_qualitative_final as Q  # noqa: E402
REV = ROOT / "output/pixray_multiseed_bs4_e50_2026-09-14/revision_2026-11/figures"
REV.mkdir(parents=True, exist_ok=True)
orig = json.loads((Q.OUT / "qualitative_final_rows.json").read_text())["main"]
Q.OUT = REV                                          # save() writes to Q.OUT
D = Q.Data(); sel = Q.select(D); R = sel["rows"]
rows = [R[k] for k in ("I1", "I2", "I3", "F1", "F2", "F3")]
# Layout only: wrap long row tags and column titles, show class names without underscores.
rows = [(t.replace(": ", ":\n").replace("Improvement (hard class)", "Improvement\n(hard class)"), a) for t, a in rows]
D.cname = {k: v.replace("_", " ") for k, v in D.cname.items()}
cols = [("03", "Parent\n(Cascade +\nAttFPN)"), ("07", "Bilayer FCN\n(best prior)\n"),
        ("08", "Parent +\nPIDray R0/R1\n"), ("09", "Ours:\ntransformer\nrefiner"), ("09M", "Ours:\nMLP\nrefiner")]
assert [m for m, _ in cols] == [m for m, _ in Q.KEY_COLS]
facts = Q.comparison_grid(D, rows, cols, "fig8_qualitative_larger_labels",
                          width=7.0, row_h=1.22, title_fs=7.2, lab_fs=7.0)
same = [(a["image_id"], a["panels"]) for a in facts] == [(b["image_id"], b["panels"]) for b in orig]
print("rows and panel values identical to published Fig. 8:", same)
for a in facts: print(a["row"], a["image_id"], a["panels"])
sys.exit(0 if same else 1)
