#!/bin/bash
# ===========================================================================
#  Dalia (HTW) SLURM batch script — INFERENCE ONLY.
#
#  This does NOT train anything. It loads an already-trained checkpoint and
#  runs examples/intuitive_physics/visualize_latent.py to produce the
#  presentation figures (latent position map, rollout breakdown, latent-vs-pixel).
#  A GPU is requested only to make the forward passes + t-SNE fast; the same
#  script runs on CPU if you drop the --gpus line.
#
#  Submit:   mkdir -p logs && sbatch viz_inference.sh [CKPT] [OUT_DIR]
#  Watch:    sq            |   log -f <JOBID>
#  Kill:     scancel <JOBID>
# ---------------------------------------------------------------------------
#  Cluster facts: partition=defq, nodes=dalianvl (4x GB200 each), compute arch
#  = aarch64. We source env.sh which sets UV_PROJECT_ENVIRONMENT to the
#  arch-specific venv ($WORK/venvs/eb_jepa_aarch64 on the compute node).
# ===========================================================================

#SBATCH --job-name=ebjepa_viz
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech      # hackathon reserved GB200 block. If jobs
                                    # won't schedule, run `scontrol show
                                    # reservation` and match the active name,
                                    # or comment this line to use the defq pool.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00             # inference is minutes; keep it short.
#SBATCH --output=logs/%x_%j.out     # logs/ must exist before sbatch
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# 0) Run from the repo root (holds env.sh, examples/, CLAUDE.md).
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# 1) Environment: env.sh picks the aarch64 venv automatically on the GPU node.
source env.sh
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# 2) Args: checkpoint, output dir, analysis selector (all | basic | cluster).
CKPT="${1:-/lustre/work/vivatech-modelusifyoucan/shared/dev_2026-06-20_04-03/exp_seed1/latest.pth.tar}"
OUT="${2:-evidence/viz}"
ANALYSIS="${3:-all}"

# 3) Prove we actually got a GPU and torch sees it BEFORE running.
echo "host=$(hostname)  arch=$(uname -m)  job=${SLURM_JOB_ID:-?}"
echo "venv=$UV_PROJECT_ENVIRONMENT"
nvidia-smi || echo "no nvidia-smi (running on CPU)"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

# 4) INFERENCE: generate the figures (PDF + PNG) from the trained weights.
echo "Visualizing CKPT=$CKPT  ->  OUT=$OUT  (analysis=$ANALYSIS)"
python -m examples.intuitive_physics.visualize_latent \
    --ckpt "$CKPT" --out "$OUT" --analysis "$ANALYSIS"

echo "DONE job=${SLURM_JOB_ID:-?}"
