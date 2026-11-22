#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

experiments=(P1 P2 P3 P4 S1 S2 S3-P S3-MBoost E1 E2 E3)
configs=(
  projects/proposal_first_pidray_occ/configs/pidray30_p1_cascade_fpn.py
  projects/proposal_first_pidray_occ/configs/pidray30_p2_cascade_attfpn.py
  projects/proposal_first_pidray_occ/configs/pidray30_p3_recall_rpn.py
  projects/proposal_first_pidray_occ/configs/pidray30_p4_cascade_rpn.py
  projects/proposal_first_pidray_occ/configs/pidray30_s1_single_decoder.py
  configs/occlusion/pidray30_s2_occ_bisam.py
  projects/proposal_first_pidray_occ/configs/pidray30_s3_proposal_first_bisam.py
  projects/proposal_first_pidray_occ/configs/pidray30_s3_mboost.py
  projects/proposal_first_pidray_occ/configs/pidray30_e1_adapter.py
  projects/proposal_first_pidray_occ/configs/pidray30_e2_hard_aware.py
  projects/proposal_first_pidray_occ/configs/pidray30_e3_conditional_bisam.py
)
work_dirs=(
  pidray30_p1 pidray30_p2 pidray30_p3 pidray30_p4 pidray30_s1
  pidray30_s2 pidray30_s3_p pidray30_s3_mboost pidray30_e1
  pidray30_e2 pidray30_e3
)

report="${PRIMARY60_REPORT:-output/reports/pidray30/pidray30_primary60.xlsx}"
log="${PRIMARY60_LOG:-output/logs/pidray30_primary60.log}"
mkdir -p "$(dirname "${report}")" "$(dirname "${log}")"
for name in "${work_dirs[@]}"; do
  directory="work_dirs/${name}"
  if [[ -e "${directory}/last_checkpoint" ]] || compgen -G "${directory}/epoch_*.pth" >/dev/null; then
    echo "STOP: ${directory} contains a checkpoint. Archive it before a fresh run." >&2
    exit 1
  fi
done
if [[ -e "${report}" ]]; then
  echo "STOP: report already exists: ${report}. Archive it before a fresh run." >&2
  exit 1
fi
if [[ -s "${log}" ]]; then
  echo "STOP: log already exists: ${log}. Archive it before a fresh run." >&2
  exit 1
fi
exec > >(tee -a "${log}") 2>&1

audit_args=()
if [[ "${XRAY_AUDIT_CONTENT_HASHES:-1}" == "1" ]]; then
  audit_args+=(--check-content-hashes)
fi
"${PYTHON_BIN}" scripts/audit_pidray.py "${audit_args[@]}"
"${PYTHON_BIN}" projects/proposal_first_pidray_occ/tools/check_static_contracts.py
"${PYTHON_BIN}" projects/proposal_first_pidray_occ/tools/check_hybrid_contracts.py

for config in "${configs[@]}"; do
  "${PYTHON_BIN}" scripts/preflight.py --config "${config}" --build-model
  "${PYTHON_BIN}" scripts/smoke_test.py "${config}" --require-retained
done

for experiment in "${experiments[@]}"; do
  "${PYTHON_BIN}" scripts/run_pidray30_experiments.py train \
    --experiments "${experiment}" --skip-preflight
done

"${PYTHON_BIN}" scripts/run_pidray30_experiments.py evaluate \
  --experiments "${experiments[@]}" \
  --splits easy hard hidden \
  --skip-preflight --continue-on-error --keep-predictions \
  --error-samples 20 --report "${report}"

mkdir -p output/reports/pidray30/qualitative
for split in easy hard hidden; do
  "${PYTHON_BIN}" scripts/generate_qualitative_comparison.py \
    --annotations "data/PIDray/annotations/test_${split}.json" \
    --image-root data/PIDray/test \
    --method "P1 Cascade-FPN=work_dirs/pidray30_p1/test/${split}/predictions.pkl" \
    --method "P2 Cascade-AttFPN=work_dirs/pidray30_p2/test/${split}/predictions.pkl" \
    --method "P3 RecallRPN=work_dirs/pidray30_p3/test/${split}/predictions.pkl" \
    --method "P4 CascadeRPN=work_dirs/pidray30_p4/test/${split}/predictions.pkl" \
    --method "S1 SingleDecoder=work_dirs/pidray30_s1/test/${split}/predictions.pkl" \
    --method "S2 Occ-BiSAM=work_dirs/pidray30_s2/test/${split}/predictions.pkl" \
    --method "S3-P GuidedBiSAM=work_dirs/pidray30_s3_p/test/${split}/predictions.pkl" \
    --method "S3-MBoost=work_dirs/pidray30_s3_mboost/test/${split}/predictions.pkl" \
    --method "E1 Adapter=work_dirs/pidray30_e1/test/${split}/predictions.pkl" \
    --method "E2 HardAware=work_dirs/pidray30_e2/test/${split}/predictions.pkl" \
    --method "E3 ConditionalBiSAM=work_dirs/pidray30_e3/test/${split}/predictions.pkl" \
    --output "output/reports/pidray30/qualitative/${split}.png" \
    --sheet-name "Qualitative ${split}" \
    --workbook "${report}"
done

find work_dirs -path '*/test/*/predictions.pkl' -delete
echo "PIDray-30 controlled workflow complete: ${report}"
