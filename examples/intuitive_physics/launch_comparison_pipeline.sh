#!/usr/bin/env bash
# launch_comparison_pipeline.sh
#
# Launches EB-JEPA, SimVP, and ConvLSTM training in parallel on SLURM, then
# submits compare.py as a dependent job that fires only when all three succeed.
#
# Usage (from repo root, after: source env.sh):
#   bash examples/intuitive_physics/launch_comparison_pipeline.sh
#
# Outputs:
#   Checkpoints in $EBJEPA_CKPTS/{intuitive_physics,simvp,convlstm}/sweep_<SWEEP>/
#   Figures + report in $EBJEPA_CKPTS/comparison_<SWEEP>/
#
set -euo pipefail
cd "$(dirname "$0")/../.."   # always run from repo root

# ── shared sweep name (predictable checkpoint paths) ────────────────────────
SWEEP="comparison_$(date +%Y-%m-%d)"
SWEEP_TAG="sweep_${SWEEP}"

echo "============================================================"
echo "  EB-JEPA vs SimVP vs ConvLSTM — full comparison pipeline"
echo "  Sweep tag : ${SWEEP_TAG}"
echo "  Repo root : $(pwd)"
echo "============================================================"

# ── predicted checkpoint paths ───────────────────────────────────────────────
JEPA_CKPT="${EBJEPA_CKPTS}/intuitive_physics/${SWEEP_TAG}/exp_seed1/latest.pth.tar"
SIMVP_CKPT="${EBJEPA_CKPTS}/intuitive_physics_simvp/${SWEEP_TAG}/exp_seed1/latest.pth.tar"
CONVLSTM_CKPT="${EBJEPA_CKPTS}/intuitive_physics_convlstm/${SWEEP_TAG}/exp_seed1/latest.pth.tar"
OUT_DIR="${EBJEPA_CKPTS}/comparison_${SWEEP}"
mkdir -p "${OUT_DIR}"

echo ""
echo "Predicted checkpoint paths:"
echo "  EB-JEPA   : ${JEPA_CKPT}"
echo "  SimVP     : ${SIMVP_CKPT}"
echo "  ConvLSTM  : ${CONVLSTM_CKPT}"
echo "  Figures   : ${OUT_DIR}"
echo ""

# ── launch EB-JEPA training ──────────────────────────────────────────────────
echo "[1/4] Submitting EB-JEPA training ..."
JEPA_OUT=$(python -m examples.launch_sbatch \
    --example intuitive_physics \
    --single \
    --sweep "${SWEEP}" \
    --fname examples/intuitive_physics/cfgs/train.yaml \
    --logging.log_wandb false \
    2>&1)
echo "${JEPA_OUT}"
JEPA_JOB=$(echo "${JEPA_OUT}" | grep -oP '(?<=Submitted job )\S+' | head -1)
echo "  → EB-JEPA job ID: ${JEPA_JOB}"

# ── launch SimVP training ─────────────────────────────────────────────────────
echo ""
echo "[2/4] Submitting SimVP training ..."
SIMVP_OUT=$(python -m examples.launch_sbatch \
    --example intuitive_physics_simvp \
    --single \
    --sweep "${SWEEP}" \
    --fname examples/intuitive_physics/cfgs/simvp_train.yaml \
    --logging.log_wandb false \
    2>&1)
echo "${SIMVP_OUT}"
SIMVP_JOB=$(echo "${SIMVP_OUT}" | grep -oP '(?<=Submitted job )\S+' | head -1)
echo "  → SimVP job ID: ${SIMVP_JOB}"

# ── launch ConvLSTM training ──────────────────────────────────────────────────
echo ""
echo "[3/4] Submitting ConvLSTM training ..."
CONVLSTM_OUT=$(python -m examples.launch_sbatch \
    --example intuitive_physics_convlstm \
    --single \
    --sweep "${SWEEP}" \
    --fname examples/intuitive_physics/cfgs/convlstm_train.yaml \
    --logging.log_wandb false \
    2>&1)
echo "${CONVLSTM_OUT}"
CONVLSTM_JOB=$(echo "${CONVLSTM_OUT}" | grep -oP '(?<=Submitted job )\S+' | head -1)
echo "  → ConvLSTM job ID: ${CONVLSTM_JOB}"

# ── submit compare job (dependent on all three) ───────────────────────────────
echo ""
echo "[4/4] Submitting comparison job (depends on all three) ..."

COMPARE_SCRIPT="${OUT_DIR}/run_compare.sh"
COMPUTE_PYTHON="${EBJEPA_WORK}/venvs/eb_jepa_aarch64/bin/python3"

cat > "${COMPARE_SCRIPT}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=compare_3models
#SBATCH --partition=${EBJEPA_SLURM_PARTITION:-defq}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30
#SBATCH --output=${OUT_DIR}/compare_%j.log
#SBATCH --reservation=${EBJEPA_SLURM_RESERVATION:-Vivatech}

set -euo pipefail
source "${EBJEPA_WORK}/eb_jepa_raphael/env.sh"
cd "${EBJEPA_WORK}/eb_jepa_raphael"

echo "=== compare.py starting at \$(date) ==="

${COMPUTE_PYTHON} -m examples.intuitive_physics.compare \
    --jepa_ckpt     "${JEPA_CKPT}" \
    --simvp_ckpt    "${SIMVP_CKPT}" \
    --convlstm_ckpt "${CONVLSTM_CKPT}" \
    --jepa_cfg      "examples/intuitive_physics/cfgs/eval.yaml" \
    --simvp_cfg     "examples/intuitive_physics/cfgs/simvp_train.yaml" \
    --convlstm_cfg  "examples/intuitive_physics/cfgs/convlstm_train.yaml" \
    --out_dir       "${OUT_DIR}" \
    --n_pairs       200

echo "=== compare.py done at \$(date) ==="
echo "Figures saved to: ${OUT_DIR}"
SBATCH

chmod +x "${COMPARE_SCRIPT}"

COMPARE_JOB=$(sbatch \
    --dependency="afterok:${JEPA_JOB}:${SIMVP_JOB}:${CONVLSTM_JOB}" \
    "${COMPARE_SCRIPT}" \
    | grep -oP '\d+$')

echo "  → Compare job ID: ${COMPARE_JOB}"

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Pipeline submitted successfully"
echo "============================================================"
echo ""
echo "  Job IDs:"
echo "    EB-JEPA training   : ${JEPA_JOB}"
echo "    SimVP   training   : ${SIMVP_JOB}"
echo "    ConvLSTM training  : ${CONVLSTM_JOB}"
echo "    Comparison         : ${COMPARE_JOB}  (runs after all three succeed)"
echo ""
echo "  Monitor:"
echo "    sq                             # queue status"
echo "    log -f ${JEPA_JOB}            # tail EB-JEPA logs"
echo "    log -f ${SIMVP_JOB}           # tail SimVP logs"
echo "    log -f ${CONVLSTM_JOB}        # tail ConvLSTM logs"
echo "    log -f ${COMPARE_JOB}         # tail comparison logs"
echo ""
echo "  Results (after compare job completes):"
echo "    ${OUT_DIR}/auroc_comparison.png"
echo "    ${OUT_DIR}/score_distributions.png"
echo "    ${OUT_DIR}/sample_predictions.png"
echo "    ${OUT_DIR}/comparison_report.txt"
echo "============================================================"
