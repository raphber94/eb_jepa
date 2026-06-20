#!/bin/bash
# =============================================================================
# TRACK 7 — full energy-based (learned-value) pipeline in ONE job. Fire-and-forget.
#
# Does, in order, with NO further interaction needed:
#   1. Train 12 epochs with the VALUE HEAD ON (saves per-epoch ckpts e-0..e-11).
#   2. Evidence plots (loss curves + collapse indicators incl. value_loss) + env gifs.
#   3. Test EACH level (1 ep=e-0, 5 ep=e-4, 12 ep=e-11) with the LEARNED-VALUE
#      planner on 10 mazes -> per-level solve gifs + success rates.
#   4. Stitch the side-by-side comparison gif across all three levels.
#
# Robustness: steps 1-2 use `set -e` (must succeed). Steps 3-4 use `set +e` so a
# planning hiccup on one level cannot destroy the training/plots already saved.
#
# Submit:  sbatch run_value_pipeline.sh
# =============================================================================
#SBATCH --job-name=ebjepa_track7
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:45:00
#SBATCH --output=track7_%j.out
#SBATCH --error=track7_%j.err

set -e

EPOCHS="${EPOCHS:-12}"
NUM_ROOMS="${NUM_ROOMS:-10}"
JOB="${SLURM_JOB_ID:-manual}"
# FIX KNOBS (default empty = original behaviour, previous results untouched):
#   SAMPLE_LEN=64  -> train value on longer windows so the hindsight goal is FAR
#                     (fixes the out-of-distribution near-goal-only value collapse)
#   VALUE_GAMMA    -> TD discount (default 0.95)
SAMPLE_LEN="${SAMPLE_LEN:-}"
VALUE_GAMMA="${VALUE_GAMMA:-0.95}"
SL_OVR=""; [ -n "$SAMPLE_LEN" ] && SL_OVR="--data.sample_length=$SAMPLE_LEN"
TAG="${TAG:-track7}"

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | Arch: $ARCH | EPOCHS=$EPOCHS | TRACK 7 (value) ==="
module load python312

WANDB_KEYFILE="${WANDB_KEYFILE:-/lustre/work/vivatech-modelusifyoucan/rbellepea/.wandb_key}"
[ -f "$WANDB_KEYFILE" ] && export WANDB_API_KEY="$(cat "$WANDB_KEYFILE")" && echo "wandb key loaded"

echo ">>> uv sync..." && uv sync --dev --project "$REPO"

RUN_DIR="$REPO/evidence/${TAG}_${JOB}"
MODEL_DIR="$RUN_DIR/train/model"
mkdir -p "$RUN_DIR"

PLAN_CFG=examples/ac_video_jepa/cfgs/planning/two_rooms/planning_mppi_value_compare.yaml
EVAL_CFG=examples/ac_video_jepa/cfgs/eval/two_rooms/eval_compare.yaml

# =============================================================================
# STEP 1 — TRAIN 12 EPOCHS WITH VALUE HEAD ON
# =============================================================================
echo ">>> [1/4] Training $EPOCHS epochs (value head ON, sample_len=${SAMPLE_LEN:-default}, gamma=$VALUE_GAMMA)..."
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml \
  --folder "$MODEL_DIR" \
  --optim.epochs="$EPOCHS" \
  --model.value_coeff=1.0 \
  --model.value_gamma="$VALUE_GAMMA" \
  --model.value_lr=0.001 \
  $SL_OVR \
  --logging.log_wandb=True \
  --meta.enable_plan_eval=False \
  --model.compile=False

# =============================================================================
# STEP 2 — EVIDENCE PLOTS + ENV GIFS
# =============================================================================
echo ">>> [2/4] Evidence plots (loss + collapse incl. value_loss) + env gifs..."
uv run --project "$REPO" python -m examples.ac_video_jepa.make_evidence \
  --metrics "$MODEL_DIR/metrics.csv" --out "$RUN_DIR"
uv run --project "$REPO" python -m examples.ac_video_jepa.explore_phase0 \
  visualize_env --out "$RUN_DIR"
echo ">>> training checkpoints:" && ls -1 "$MODEL_DIR"/*.pth.tar | sed 's#.*/##' | tr '\n' ' '; echo

# =============================================================================
# STEP 3 — TEST EACH LEVEL WITH THE LEARNED-VALUE PLANNER (non-fatal)
# =============================================================================
set +e   # a planning failure on one level must NOT abort the whole job
declare -A CKPT=( [1epoch]=e-0 [5epoch]=e-4 [12epoch]=e-11 )
MODEL_DIRS=(); LABELS=()
for tag in 1epoch 5epoch 12epoch; do
    src="$MODEL_DIR/${CKPT[$tag]}.pth.tar"
    if [ ! -f "$src" ]; then echo "!! [3/4] missing $src, skipping $tag"; continue; fi
    snapdir="$RUN_DIR/eval/$tag/model"
    mkdir -p "$snapdir"
    cp "$src" "$snapdir/latest.pth.tar"
    echo ">>> [3/4] Learned-value planning eval on $tag ($src) ..."
    uv run --project "$REPO" python -m examples.ac_video_jepa.main \
        --fname examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml \
        --meta.model_folder "$snapdir" \
        --meta.eval_only_mode True \
        --meta.skip_unroll_eval True \
        --logging.log_wandb False \
        --model.compile False \
        --model.value_coeff=1.0 \
        --eval.plan_cfg_path "$PLAN_CFG" \
        --eval.eval_cfg_path "$EVAL_CFG"
    if [ $? -eq 0 ]; then
        MODEL_DIRS+=("$snapdir"); LABELS+=("${tag/epoch/ ep value}")
        echo ">>> [3/4] $tag eval OK"
    else
        echo "!! [3/4] $tag eval FAILED (continuing)"
    fi
done

# =============================================================================
# STEP 4 — STITCH THE SIDE-BY-SIDE COMPARISON GIF
# =============================================================================
if [ "${#MODEL_DIRS[@]}" -ge 2 ]; then
    OUT="$RUN_DIR/solve_comparison_value.gif"
    MODELS_CSV=$(IFS=,; echo "${MODEL_DIRS[*]}")
    LABELS_CSV=$(IFS=,; echo "${LABELS[*]}")
    echo ">>> [4/4] Stitching side-by-side comparison: $OUT"
    uv run --project "$REPO" python -m examples.ac_video_jepa.compare_gifs \
        --models "$MODELS_CSV" --labels "$LABELS_CSV" \
        --out "$OUT" --num_rooms "$NUM_ROOMS"
else
    echo "!! [4/4] fewer than 2 levels evaluated — skipping comparison gif"
fi

echo "=================================================================="
echo "=== TRACK 7 PIPELINE DONE. Everything under: $RUN_DIR"
echo "===   train/model/*.pth.tar     : 1/5/12-epoch value models (e-0,e-4,e-11)"
echo "===   loss_curves.png, collapse_indicators.png, summary.txt"
echo "===   dataset_traj.gif, env_rollout.gif, task_definition.png"
echo "===   eval/<level>/model/plan_eval/.../ep_*/agent_steps_*.gif"
echo "===   solve_comparison_value.gif : the side-by-side across all levels"
echo "=================================================================="
ls -la "$RUN_DIR"