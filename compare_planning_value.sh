#!/bin/bash
# =============================================================================
# Track 7 side-by-side: "which VALUE-trained snapshot solves the maze" using the
# LEARNED-VALUE (energy-based) planner — the value analogue of compare_planning.sh.
#
# Takes the 12-epoch value run's per-epoch checkpoints (e-0=1ep, e-4=5ep,
# e-11=12ep), evaluates each with learned_value MPPI over the same 10 rooms, and
# stitches the side-by-side gif.
#
# Point it at the value run's model dir (auto-detects newest if omitted):
#     sbatch compare_planning_value.sh
#     VRUN=evidence/value12_<jobid>/model sbatch compare_planning_value.sh
#
# Validate one snapshot first:  SNAPS="12epoch" sbatch compare_planning_value.sh
# =============================================================================
#SBATCH --job-name=ebjepa_valcmp
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=valcmp_%j.out
#SBATCH --error=valcmp_%j.err

set -e

SNAPS="${SNAPS:-1epoch 5epoch 12epoch}"
NUM_ROOMS="${NUM_ROOMS:-10}"

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
echo "=== Host: $(hostname) | snaps: $SNAPS ==="
echo ">>> uv sync..." && uv sync --dev --project "$REPO"

# Locate the value training run (newest value12_* unless VRUN given).
VRUN="${VRUN:-$(ls -dt "$REPO"/evidence/value12_*/model 2>/dev/null | head -1)}"
if [ -z "$VRUN" ]; then echo "!! no value run found (evidence/value12_*/model)"; exit 1; fi
echo ">>> Value run: $VRUN"

PLAN_CFG=examples/ac_video_jepa/cfgs/planning/two_rooms/planning_mppi_value_compare.yaml
EVAL_CFG=examples/ac_video_jepa/cfgs/eval/two_rooms/eval_compare.yaml

# Map epoch-count -> per-epoch checkpoint file (e-(N-1)).
declare -A CKPT=( [1epoch]=e-0 [5epoch]=e-4 [12epoch]=e-11 )

BASE="$REPO/evidence/valcmp_${SLURM_JOB_ID:-manual}"
MODEL_DIRS=()
LABELS=()
for tag in $SNAPS; do
    src="$VRUN/${CKPT[$tag]}.pth.tar"
    if [ ! -f "$src" ]; then echo "!! missing $src, skipping $tag"; continue; fi
    snapdir="$BASE/$tag/model"
    mkdir -p "$snapdir"
    cp "$src" "$snapdir/latest.pth.tar"   # main.py eval loads <folder>/latest.pth.tar
    echo ">>> Learned-value planning eval on $tag ($src)"
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
    MODEL_DIRS+=("$snapdir")
    LABELS+=("${tag/epoch/ ep value}")
done

if [ "${#MODEL_DIRS[@]}" -ge 2 ]; then
    OUT="$BASE/solve_comparison_value.gif"
    MODELS_CSV=$(IFS=,; echo "${MODEL_DIRS[*]}")
    LABELS_CSV=$(IFS=,; echo "${LABELS[*]}")
    echo ">>> Stitching learned-value side-by-side: $OUT"
    uv run --project "$REPO" python -m examples.ac_video_jepa.compare_gifs \
        --models "$MODELS_CSV" --labels "$LABELS_CSV" \
        --out "$OUT" --num_rooms "$NUM_ROOMS"
    echo "=== Done. Value comparison gif: $OUT ==="
else
    echo "=== Single-snapshot validation done. Gifs under ${MODEL_DIRS[0]}/plan_eval/ ==="
fi
