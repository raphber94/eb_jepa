#!/bin/bash
# =============================================================================
# Phase-0 smoke test: train the Two Rooms AC-Video-JEPA for ONE epoch on GPU.
#
# Purpose: confirm the world model trains cleanly and does NOT collapse out of
# the gate, before committing to the full 12-epoch run. Watch the anti-collapse
# signals in the log (pred & idm should fall; std low & not climbing; sim_t low
# but nonzero; probe_loss falling).
#
# Models on slurm_test.sh (same partition/reservation/env.sh pattern that
# already ran successfully as job 74987 on the aarch64 GB200 node).
#
# Submit from the repo root:   sbatch train_1epoch.sh
# Watch:                       tail -f train1ep_<jobid>.out
# Status:                      squeue -u $USER
# =============================================================================
#SBATCH --job-name=ebjepa_1epoch
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
# No hardcoded --account: SLURM uses your DEFAULT account (team allocation).
# Override with: sbatch --account=<acct> train_1epoch.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=train1ep_%j.out
#SBATCH --error=train1ep_%j.err

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"

echo "=== Host: $(hostname) | Arch: $ARCH | Date: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a') ==="
echo "=== venv: $UV_PROJECT_ENVIRONMENT ==="

module load python312
echo "=== Python: $(python3 --version) ==="

# --- WANDB AUTH ---------------------------------------------------------------
# Read the API key from a file OUTSIDE the repo (never commit secrets). Override
# the path with WANDB_KEYFILE=... if you store it elsewhere.
WANDB_KEYFILE="${WANDB_KEYFILE:-/lustre/work/vivatech-modelusifyoucan/rbellepea/.wandb_key}"
if [ -f "$WANDB_KEYFILE" ]; then
    export WANDB_API_KEY="$(cat "$WANDB_KEYFILE")"
    echo "=== wandb: key loaded from $WANDB_KEYFILE (project=eb_jepa) ==="
else
    echo "=== wandb: no key file at $WANDB_KEYFILE — run will fail if log_wandb=True ==="
fi

# Ensure deps are present for this arch (fast no-op if already synced).
echo ">>> uv sync..."
uv sync --dev --project "$REPO"

# --- EVIDENCE DIR -------------------------------------------------------------
# Everything for this run (checkpoints, metrics.csv, plots, gifs, summary) lands
# in one self-contained dir on /lustre, visible from the login node + IDE.
RUN_DIR="$REPO/evidence/1epoch_${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR"
echo ">>> Evidence dir: $RUN_DIR"

# --- ONE-EPOCH SMOKE TEST -----------------------------------------------------
# Overrides (dot notation -> load_config):
#   optim.epochs=1          one pass (~100000//384 ≈ 260 steps)
#   logging.log_wandb=False no wandb login needed for the smoke test
#   meta.enable_plan_eval=False  skip the slow planning eval; we want the epoch + loss curves
#   model.compile=False     skip torch.compile warmup so it starts fast
#   --folder                write checkpoints + metrics.csv into the evidence dir
echo ">>> Training 1 epoch..."
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml \
  --folder "$RUN_DIR/model" \
  --optim.epochs=1 \
  --logging.log_wandb=True \
  --meta.enable_plan_eval=False \
  --model.compile=False

# --- EVIDENCE: loss curves + anti-collapse indicator plots + verdict ----------
echo ">>> Building evidence plots..."
uv run --project "$REPO" python -m examples.ac_video_jepa.make_evidence \
  --metrics "$RUN_DIR/model/metrics.csv" --out "$RUN_DIR"

# --- EVIDENCE: environment gifs (ball/wall/door + a sample trajectory) ---------
echo ">>> Rendering environment gifs..."
uv run --project "$REPO" python -m examples.ac_video_jepa.explore_phase0 \
  visualize_env --out "$RUN_DIR"

echo "=== Done. Evidence bundle in: $RUN_DIR ==="
ls -la "$RUN_DIR"
