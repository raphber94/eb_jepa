#!/usr/bin/env bash
# run_compare.sh — 3-model comparison, auto-detects checkpoints
#
# Usage (from repo root, venv already activated):
#   source env.sh
#   source $UV_PROJECT_ENVIRONMENT/bin/activate
#   bash run_compare.sh
#
set -euo pipefail
cd /lustre/work/vivatech-modelusifyoucan/jtoudjine/eb_jepa_raphael

CKPTS=/lustre/work/vivatech-modelusifyoucan/jtoudjine/checkpoints

# ── hardcoded JEPA + SimVP (from the run on 2026-06-20) ─────────────────────
JEPA_CKPT="${CKPTS}/intuitive_physics/dev_2026-06-20_13-36/exp_seed1/latest.pth.tar"
SIMVP_CKPT="${CKPTS}/intuitive_physics_simvp/dev_2026-06-20_13-36/exp_seed1/latest.pth.tar"

# ── auto-detect the most recent ConvLSTM checkpoint ─────────────────────────
CONVLSTM_CKPT=$(find "${CKPTS}/intuitive_physics_convlstm" -name "latest.pth.tar" \
    | sort | tail -1)

if [[ -z "${CONVLSTM_CKPT}" ]]; then
    echo "ERROR: No ConvLSTM checkpoint found in ${CKPTS}/intuitive_physics_convlstm/"
    echo "  → Launch training first: python -m examples.launch_sbatch --example intuitive_physics_convlstm --single --logging.log_wandb false"
    exit 1
fi

OUT_DIR="${CKPTS}/comparison_3models_$(date +%Y-%m-%d)"
mkdir -p "${OUT_DIR}"

echo "=== 3-model comparison ==="
echo "  EB-JEPA  : ${JEPA_CKPT}"
echo "  SimVP    : ${SIMVP_CKPT}"
echo "  ConvLSTM : ${CONVLSTM_CKPT}"
echo "  Out dir  : ${OUT_DIR}"
echo ""

python -m examples.intuitive_physics.compare \
    --jepa_ckpt     "${JEPA_CKPT}" \
    --simvp_ckpt    "${SIMVP_CKPT}" \
    --convlstm_ckpt "${CONVLSTM_CKPT}" \
    --out_dir       "${OUT_DIR}"

echo ""
echo "=== Copying figures to repo ==="
cp "${OUT_DIR}/auroc_comparison.png"   examples/intuitive_physics/
cp "${OUT_DIR}/score_distributions.png" examples/intuitive_physics/
cp "${OUT_DIR}/sample_predictions.png" examples/intuitive_physics/
cp "${OUT_DIR}/comparison_report.txt"  examples/intuitive_physics/

echo ""
echo "Done. Results in:"
echo "  ${OUT_DIR}/"
echo "  examples/intuitive_physics/"
