#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh

experiments=(P1 P2 P3 P4 S1 S2 S3-P S3-MBoost E1 E2 E3)
configs=(
  projects/proposal_first_pidray_occ/configs/pixray_p1_cascade_fpn.py
  projects/proposal_first_pidray_occ/configs/pixray_p2_cascade_attfpn.py
  projects/proposal_first_pidray_occ/configs/pixray_p3_cascade_attfpn_recall_rpn.py
  projects/proposal_first_pidray_occ/configs/pixray_p4_enhanced_proposal_learning.py
  projects/proposal_first_pidray_occ/configs/pixray_s1_single_decoder.py
  configs/occlusion/pixray_s2_occ_bisam.py
  projects/proposal_first_pidray_occ/configs/pixray_s3_proposal_first_bisam.py
  projects/proposal_first_pidray_occ/configs/pixray_s3_mboost.py
  projects/proposal_first_pidray_occ/configs/pixray_e1_adapter.py
  projects/proposal_first_pidray_occ/configs/pixray_e2_hard_aware.py
  projects/proposal_first_pidray_occ/configs/pixray_e3_conditional_bisam.py
)
work_dirs=(
  pixray_p1 pixray_p2 pixray_p3 pixray_p4 pixray_s1 pixray_s2
  pixray_s3_p pixray_s3_mboost pixray_e1 pixray_e2 pixray_e3
)

report="${PIXRAY_PRIMARY60_REPORT:-output/reports/pixray/pixray_primary60.xlsx}"
log="${PIXRAY_PRIMARY60_LOG:-output/logs/pixray_primary60.log}"
mkdir -p "$(dirname "${report}")" "$(dirname "${log}")"

# Never overwrite or continue an earlier training run.
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

echo "PIXray controlled protocol: 11 experiments, 60 epochs, fresh checkpoints"
echo "physical_batch=${XRAY_BATCH_SIZE_PER_GPU} effective_batch=${XRAY_EFFECTIVE_BATCH_SIZE}"

audit_args=()
if [[ "${XRAY_AUDIT_CONTENT_HASHES:-1}" == "1" ]]; then
  audit_args+=(--check-content-hashes)
fi
"${PYTHON_BIN}" scripts/audit_pixray.py "${audit_args[@]}"

for config in "${configs[@]}"; do
  "${PYTHON_BIN}" scripts/preflight_pixray.py --config "${config}" --build-model
done

if [[ "${XRAY_SKIP_SMOKE_TESTS:-0}" != "1" ]]; then
  for config in "${configs[@]}"; do
    smoke_args=()
    case "${config}" in
      *pixray_p1_*|*pixray_p2_*|*pixray_p3_*|*pixray_p4_*) ;;
      *) smoke_args+=(--require-retained) ;;
    esac
    "${PYTHON_BIN}" scripts/smoke_test.py "${config}" "${smoke_args[@]}"
  done
fi

for experiment in "${experiments[@]}"; do
  "${PYTHON_BIN}" scripts/run_pixray_experiments.py train \
    --experiments "${experiment}" --skip-preflight
done

"${PYTHON_BIN}" scripts/run_pixray_experiments.py evaluate \
  --experiments "${experiments[@]}" --skip-preflight \
  --continue-on-error --keep-predictions --error-samples 20 \
  --report "${report}"

mkdir -p output/reports/pixray/qualitative
methods=()
for index in "${!experiments[@]}"; do
  methods+=(--method "${experiments[$index]}=work_dirs/${work_dirs[$index]}/test/test/predictions.pkl")
done
"${PYTHON_BIN}" scripts/generate_qualitative_comparison.py \
  --annotations data/PIXray/annotations/test.json \
  --image-root data/PIXray/test \
  "${methods[@]}" \
  --output output/reports/pixray/qualitative/test.png \
  --sheet-name "PIXray Qualitative" --workbook "${report}"

find work_dirs -path '*/test/test/predictions.pkl' -delete
echo "PIXray controlled workflow complete: ${report}"
