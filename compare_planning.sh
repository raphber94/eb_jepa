#!/bin/bash
# =============================================================================
# Side-by-side "which model solves the maze" comparison.
#
# Runs distance (repr_dist) MPPI planning on the 1/5/12-epoch checkpoints over
# the SAME 10 rooms (env room generator is deterministic), then stitches the
# per-room gifs into one side-by-side GIF.
#
# Validate on ONE model first (fast, confirms the planning path works):
#     MODELS_TAGS=12epoch sbatch compare_planning.sh
# Full 3-way comparison:
#     sbatch compare_planning.sh
# =============================================================================
#SBATCH --job-name=ebjepa_compare
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=compare_%j.out
#SBATCH --error=compare_%j.err

set -e

MODELS_TAGS="${MODELS_TAGS:-1epoch 5epoch 12epoch}"   # which runs to compare
NUM_ROOMS="${NUM_ROOMS:-10}"

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
echo "=== Host: $(hostname) | Arch: $ARCH | tags: $MODELS_TAGS ==="
module load python312
echo ">>> uv sync..." && uv sync --dev --project "$REPO"

PLAN_CFG=examples/ac_video_jepa/cfgs/planning/two_rooms/planning_mppi_compare.yaml
EVAL_CFG=examples/ac_video_jepa/cfgs/eval/two_rooms/eval_compare.yaml

# Resolve each tag to its most-recent evidence/<tag>_*/model dir.
MODEL_DIRS=()
LABELS=()
for tag in $MODELS_TAGS; do
    d=$(ls -dt "$REPO"/evidence/${tag}_*/model 2>/dev/null | head -1)
    if [ -z "$d" ]; then echo "!! no checkpoint for $tag, skipping"; continue; fi
    MODEL_DIRS+=("$d")
    LABELS+=("${tag/epoch/ epoch}")   # "12epoch" -> "12 epoch"
    echo ">>> Planning eval on $tag: $d"
    uv run --project "$REPO" python -m examples.ac_video_jepa.main \
        --meta.model_folder "$d" \
        --meta.eval_only_mode True \
        --meta.skip_unroll_eval True \
        --logging.log_wandb False \
        --model.compile False \
        --eval.plan_cfg_path "$PLAN_CFG" \
        --eval.eval_cfg_path "$EVAL_CFG"
done

# Stitch only if we ran more than one model.
if [ "${#MODEL_DIRS[@]}" -ge 2 ]; then
    OUT="$REPO/evidence/compare_${SLURM_JOB_ID:-manual}/solve_comparison.gif"
    MODELS_CSV=$(IFS=,; echo "${MODEL_DIRS[*]}")
    LABELS_CSV=$(IFS=,; echo "${LABELS[*]}")
    echo ">>> Stitching side-by-side: $OUT"
    uv run --project "$REPO" python -m examples.ac_video_jepa.compare_gifs \
        --models "$MODELS_CSV" --labels "$LABELS_CSV" \
        --out "$OUT" --num_rooms "$NUM_ROOMS"
    echo "=== Done. Comparison gif: $OUT ==="
else
    echo "=== Single-model validation done. Check plan_eval gifs under ${MODEL_DIRS[0]}/plan_eval/ ==="
fi