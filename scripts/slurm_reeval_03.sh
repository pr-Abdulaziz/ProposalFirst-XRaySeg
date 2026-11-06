#!/usr/bin/env bash
# Re-evaluate Method 03's epoch-60 checkpoint through the SAME inline evaluator
# used for 09 and 09M, to close the evaluation-provenance asymmetry found by the
# 2026-09-12 fairness audit.
#
# No retraining. Writes to a NEW directory; the original predictions.pkl and
# checkpoint, whose hashes the 2026-09-11 manifest records, are preserved.
#SBATCH --exclude=issnode1,issnode2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --output=output/audit/2026-09-12-fairness/reeval03_%j.log
set -uo pipefail
cd /nfs/ahalqahtani/ProposalFirst-XRaySeg
source scripts/common.sh
export PYTHONUNBUFFERED=1
export XRAY_NUM_WORKERS=2

CONFIG=projects/proposal_first_pidray_occ/configs/pixray_p2_cascade_attfpn.py
CKPT=work_dirs/pixray_p2/epoch_60.pth
OUTDIR=work_dirs/pixray_p2/test_reeval_2026-09-12

echo "node=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"

# Preservation guards: the original artefacts are referenced by the result
# manifest and must survive this run untouched.
if [[ -e "${OUTDIR}" ]]; then
  echo "ABORT: ${OUTDIR} already exists; refusing to overwrite a previous re-evaluation"
  exit 2
fi
case "${OUTDIR}" in
  work_dirs/pixray_p2/test|work_dirs/pixray_p2/test/*)
    echo "ABORT: output directory would overwrite the original evaluation"; exit 2;;
esac

"${PYTHON_BIN}" - <<'GUARD' || exit 3
import hashlib, json
from pathlib import Path
m = json.loads(Path("output/audit/2026-09-11/result_manifest.json").read_text())
e = [x for x in m["methods"] if str(x.get("id")) == "03"][0]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
ok = True
for label, path, expected in (
        ("checkpoint", e["checkpoint"], e["checkpoint_sha256"]),
        ("predictions", e["predictions"], e["predictions_sha256"]),
        ("test.json", m["annotations"], m["annotation_sha256"])):
    got = sha(path)
    print(f"{'PASS' if got == expected else 'FAIL'} {label} {path}")
    ok &= got == expected
raise SystemExit(0 if ok else 1)
GUARD

mkdir -p "${OUTDIR}"
"${PYTHON_BIN}" tools/test.py "${CONFIG}" "${CKPT}" \
  --work-dir "${OUTDIR}" --out "${OUTDIR}/predictions.pkl"
status=$?

echo "--- originals must be unchanged after the run ---"
"${PYTHON_BIN}" - <<'VERIFY'
import hashlib, json
from pathlib import Path
m = json.loads(Path("output/audit/2026-09-11/result_manifest.json").read_text())
e = [x for x in m["methods"] if str(x.get("id")) == "03"][0]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
for label, path, expected in (("checkpoint", e["checkpoint"], e["checkpoint_sha256"]),
                              ("predictions", e["predictions"], e["predictions_sha256"])):
    print(f"{'PRESERVED' if sha(path) == expected else 'MODIFIED!!'} {label} {path}")
print(f"manifest offline numbers to compare against: "
      f"bbox AP {e['results']['bbox']['official']['AP']:.10f}  "
      f"mask AP {e['results']['segm']['official']['AP']:.10f}")
VERIFY
echo "reeval exit=${status}  end=$(date -Is)"
exit "${status}"
