#!/bin/bash
# Quick value-head diagnostic: is the trained value INFORMATIVE or COLLAPSED?
# Tests the 12-epoch value model from the track7 run in both the near (17) and
# far (64) goal regime. Read-only — loads checkpoints, evaluates nothing on env.
#SBATCH --job-name=ebjepa_valdiag
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=valdiag_%j.out
#SBATCH --error=valdiag_%j.err

set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv sync --dev --project "$REPO" >/dev/null

CKPT="${CKPT:-$REPO/evidence/track7_75937/train/model/e-11.pth.tar}"
echo ">>> Diagnosing value head: $CKPT"
for SL in 17 64; do
    uv run --project "$REPO" python -m examples.ac_video_jepa.value_diagnostic \
        --ckpt "$CKPT" --sample_length "$SL"
done
echo "=== diagnostic done ==="
