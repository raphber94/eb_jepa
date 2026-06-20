#!/bin/bash
# =============================================================================
# Phase-0 calibration run: train the Two Rooms AC-Video-JEPA for N epochs (def 5).
#
# Why 5: gives the epoch-5 probe-XY-MSE + per-term collapse numbers we use to
# calibrate the pruning gate before committing to the full 12-epoch sweep, and a
# longer window to confirm the anti-collapse equilibrium holds (not just survives
# the first epoch).
#
# Override epoch count without editing:   EPOCHS=8 sbatch train_5epoch.sh
# Submit from the repo root:              sbatch train_5epoch.sh
# Watch:                                  tail -f train5ep_<jobid>.out
# =============================================================================
#SBATCH --job-name=ebjepa_5epoch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=train5ep_%j.out
#SBATCH --error=train5ep_%j.err

set -e

EPOCHS="${EPOCHS:-5}"          # number of epochs (override: EPOCHS=8 sbatch ...)

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"

echo "=== Host: $(hostname) | Arch: $ARCH | Date: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a') ==="
echo "=== venv: $UV_PROJECT_ENVIRONMENT | EPOCHS=$EPOCHS ==="

module load python312
echo "=== Python: $(python3 --version) ==="

# --- WANDB AUTH ---------------------------------------------------------------
WANDB_KEYFILE="${WANDB_KEYFILE:-/lustre/work/vivatech-modelusifyoucan/rbellepea/.wandb_key}"
if [ -f "$WANDB_KEYFILE" ]; then
    export WANDB_API_KEY="$(cat "$WANDB_KEYFILE")"
    echo "=== wandb: key loaded from $WANDB_KEYFILE (project=eb_jepa) ==="
else
    echo "=== wandb: no key file at $WANDB_KEYFILE — run will fail if log_wandb=True ==="
fi

echo ">>> uv sync..."
uv sync --dev --project "$REPO"

# --- EVIDENCE DIR -------------------------------------------------------------
RUN_DIR="$REPO/evidence/${EPOCHS}epoch_${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR"
echo ">>> Evidence dir: $RUN_DIR"

# --- TRAIN --------------------------------------------------------------------
# compile=False: stay on the path proven by the 1-epoch run (5 epochs ≈ 3-4 min,
# so torch.compile's warm-up buys little). The full run can flip compile=True.
# enable_plan_eval stays False (planning eval is a later phase); probe_loss in
# metrics.csv is our cheap quality/collapse proxy for now.
echo ">>> Training $EPOCHS epochs..."
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml \
  --folder "$RUN_DIR/model" \
  --optim.epochs="$EPOCHS" \
  --logging.log_wandb=True \
  --meta.enable_plan_eval=False \
  --model.compile=False

# --- EVIDENCE: loss curves + anti-collapse indicator plots + verdict ----------
echo ">>> Building evidence plots..."
uv run --project "$REPO" python -m examples.ac_video_jepa.make_evidence \
  --metrics "$RUN_DIR/model/metrics.csv" --out "$RUN_DIR"

# --- EVIDENCE: environment gifs -----------------------------------------------
echo ">>> Rendering environment gifs..."
uv run --project "$REPO" python -m examples.ac_video_jepa.explore_phase0 \
  visualize_env --out "$RUN_DIR"

echo "=== Done. Evidence bundle in: $RUN_DIR ==="
ls -la "$RUN_DIR"
