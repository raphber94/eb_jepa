#!/bin/bash
# =============================================================================
# Track 7: 12-epoch training with the VALUE HEAD ON (energy-based / learned value).
#
# Same world model + anti-collapse regularizer as the distance baseline, but with
# value_coeff>0 so the GoalValueHead is trained by TD on the model's own rollouts.
# Per-epoch checkpoints (e-0..e-11) are saved automatically -> the 1/5/12-epoch
# snapshots we later plan with are e-0, e-4, e-11.
#
# Submit from the repo root:   sbatch train_value.sh
# =============================================================================
#SBATCH --job-name=ebjepa_value
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=trainval_%j.out
#SBATCH --error=trainval_%j.err

set -e

EPOCHS="${EPOCHS:-12}"

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | Arch: $ARCH | EPOCHS=$EPOCHS | VALUE HEAD ON ==="
module load python312

WANDB_KEYFILE="${WANDB_KEYFILE:-/lustre/work/vivatech-modelusifyoucan/rbellepea/.wandb_key}"
[ -f "$WANDB_KEYFILE" ] && export WANDB_API_KEY="$(cat "$WANDB_KEYFILE")" && echo "wandb key loaded"

echo ">>> uv sync..." && uv sync --dev --project "$REPO"

RUN_DIR="$REPO/evidence/value${EPOCHS}_${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR"
echo ">>> Evidence dir: $RUN_DIR"

# Value-head overrides (read by main.py via cfg.model.get):
#   value_coeff=1.0   gates the whole TD value-training path ON
#   value_gamma=0.95  TD discount
#   value_lr=1e-3     value-head optimizer LR
echo ">>> Training $EPOCHS epochs with value head ON..."
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml \
  --folder "$RUN_DIR/model" \
  --optim.epochs="$EPOCHS" \
  --model.value_coeff=1.0 \
  --model.value_gamma=0.95 \
  --model.value_lr=1.e-3 \
  --logging.log_wandb=True \
  --meta.enable_plan_eval=False \
  --model.compile=False

echo ">>> Building evidence plots (now incl. value_loss)..."
uv run --project "$REPO" python -m examples.ac_video_jepa.make_evidence \
  --metrics "$RUN_DIR/model/metrics.csv" --out "$RUN_DIR"

echo ">>> Rendering environment gifs..."
uv run --project "$REPO" python -m examples.ac_video_jepa.explore_phase0 \
  visualize_env --out "$RUN_DIR"

echo "=== Done. Value run evidence: $RUN_DIR ==="
echo "    snapshots: e-0 (1 epoch), e-4 (5 epoch), e-11 (12 epoch)"
ls -la "$RUN_DIR/model"/*.pth.tar
