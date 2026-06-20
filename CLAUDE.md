# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Hackathon context

This is the **EB-JEPA** library (Energy-Based Joint-Embedding Predictive Architectures) used during the VivaTech 24h hackathon. The active work targets **Track 10 — Does intuitive physics emerge?**: feed a trained video-JEPA matched plausible/impossible Moving-MNIST clips and measure whether `predcost` (latent prediction energy) is higher on the impossible ones (violation-of-expectation signal). The example is at `examples/intuitive_physics/`.

---

## Environment setup (HTW cluster)

Everything must live under `/lustre/work` — the home directory quota is too small for git, venvs, and caches.

```bash
source env.sh           # sets all paths; add to ~/.bashrc
```

Key env vars set by `env.sh`:

| Variable | Default | Purpose |
|---|---|---|
| `EBJEPA_DSETS` | `$WORK/datasets` | Dataset root (Moving MNIST auto-downloads here) |
| `EBJEPA_CKPTS` | `$WORK/checkpoints` | Checkpoint root |
| `WANDB_DISABLED` | `false` | Set `true` to silence W&B |
| `UV_PROJECT_ENVIRONMENT` | `$WORK/venvs/eb_jepa_<arch>` | venv (arch-specific for x86/aarch64) |

---

## Commands

source $UV_PROJECT_ENVIRONMENT/bin/activate -> already set up env
The code will need to be run on a node cluster of GPUs : 

Do not run the training code by yourself, I will run it by myself on the adapted GPUs


---

## Architecture overview

### Library core (`eb_jepa/`)

**`jepa.py`** — the central abstraction:
- `JEPAbase`: encoder + action_encoder + predictor. Provides `encode()`.
- `JEPA(JEPAbase)`: adds `regularizer` (anti-collapse) and `predcost` (prediction loss). The key method is `unroll()`.
- `JEPAProbe`: frozen JEPA + trainable head (used for detection / pixel decoding).

**`unroll()` in detail** — both training and inference go through this:
```python
# Training (parallel mode): predicts all timesteps, refeeds GT context on left
_, (total_loss, reg_loss, _, reg_loss_dict, pred_loss) = jepa.unroll(
    x, actions=None, nsteps=K, unroll_mode="parallel", compute_loss=True)

# Inference (return per-step predictions as a list)
preds, _ = jepa.unroll(x, actions=None, nsteps=K,
                        unroll_mode="parallel", compute_loss=False, return_all_steps=True)
# preds[k] has shape [B, D, T-context_length, H', W']
```
The batch-mean `predcost` (5th element of the loss tuple) is `SquareLossSeq` — MSE in projected latent space. For per-clip energy, you must NOT use batch reduction.

**`architectures.py`** — building blocks:
- `ResNet5`: lightweight 2-block ResNet; handles 4D `[B,C,H,W]` and 5D `[B,C,T,H,W]` via `TemporalBatchMixin` (processes time as extra batch dim).
- `ResUNet`: UNet with residual skip connections; same 4D/5D handling. Used as the predictor.
- `StateOnlyPredictor`: wraps `ResUNet`, concatenates `state[t]` and `state[t+1]` on the channel dim before forwarding.
- `ImpalaEncoder` / `RNNPredictor`: used only by `ac_video_jepa`.
- `Projector`: MLP built from a spec string like `"64-256-256"`.

**`losses.py`**:
- `VCLoss`: VICReg regularizer (std hinge + covariance). Used as `regularizer`.
- `SquareLossSeq`: MSE through a projector. Used as `predcost`.
- `VC_IDM_Sim_Regularizer`: extended regularizer for `ac_video_jepa` (adds temporal similarity + inverse dynamics losses).

### Examples pattern

Each example is self-contained with `main.py` (training) + `eval.py` + `cfgs/`. All use the same `eb_jepa` core with different architectures / datasets.

### Track 10 — `examples/intuitive_physics/`

| File | Role |
|---|---|
| `stimuli.py` | Dataset generator. `ProceduralBouncingMNIST` for training. `build_probe_pairs()` for held-out matched pairs. |
| `main.py` | Training loop. Reuses `build_jepa()` factory (ResNet5 + ResUNet). |
| `eval.py` | **`clip_energy()` is the main TODO.** Everything else (loading, AUROC, reporting) is provided. |
| `cfgs/train.yaml` | Training config. |
| `cfgs/eval.yaml` | Eval config (probe params: `n_pairs`, `seed`, `T`). |

**Stimulus format:** video tensors are `[B, 1, T, 64, 64]` in `[0,1]`. The `ProceduralBouncingMNIST` dataset returns `{"video": ..., "digit_location": ...}`.

**Three violation types** (from `stimuli.VIOLATIONS`):
- `teleport`: position jumps ≥18px at frame `t_v`
- `reversal`: velocity instantly negates in free flight
- `passthrough`: digit wraps through wall (modular arithmetic)

Pairs share frames `0..t_v-1` exactly — bias control is built-in.

---

## Key hyperparameters (Track 10)

**Do not change these** — they are the validated optimal configuration and any deviation is a confound:

```yaml
loss.cov_coeff: 100     # covariance regularization weight
loss.std_coeff: 10      # variance (std) regularization weight
model.steps: 4          # K prediction steps (performance saturates beyond K=8)
```

**Collapse watch:** if `train/std_loss` stays high while `train/pred_loss` → 0, the encoder is collapsing. All `predcost` values become meaningless. Mitigation: raise `std_coeff` or lower LR.

---

## Implementing `clip_energy` (the core TODO)

The function in `examples/intuitive_physics/eval.py` must return a `[N]` tensor (one energy per clip). Key points:

1. Encode: `state = jepa.encoder(clips)` → `[B, D, T, H, W]`
2. Run the parallel-mode unroll loop K steps (see `JEPA.unroll` source at `eb_jepa/jepa.py`), accumulating `jepa.predcost(state, pred)` at each step.
3. **Reduce per-clip** (mean over D, T, H, W but NOT over the batch dim) so individual clips can be compared.
4. Sanity check: `jepa.unroll(..., compute_loss=True)` returns the batch-mean pred loss as element 5 — your per-clip energies should average to it.

The `predcost` field on JEPA is `SquareLossSeq`, which internally projects then computes MSE. To get per-clip reduction you need to bypass its `reduction="mean"` default — look at `sq_loss` in `losses.py` with `reduction="none"` or compute the loss manually without the projector for a simpler approximation.

---

## Checkpoint structure

```
$EBJEPA_CKPTS/
└── {example_name}/
    └── {sweep_name}/        # e.g. dev_2026-06-20_14-30 or sweep_...
        └── {exp_name}_seed{N}/
            ├── latest.pth.tar      # always overwritten
            ├── epoch_{N}.pth.tar   # saved every save_every epochs
            └── wandb_run_id.txt
```

Load a checkpoint: `load_checkpoint(ckpt_path, jepa, optimizer=None, device=device)` from `eb_jepa.training_utils`.

---

## Cluster notes (DALIA / HTW)

- **Compute nodes:** GB200 (aarch64, 185 GB VRAM/GPU). Login node is x86_64 — the venv is arch-specific.
- **Partition:** `defq`, 2-day limit. Default SLURM job: 1 GPU, 8 CPUs.
- **SLURM launcher:** `examples/launch_sbatch.py` via `submitit`. Handles seeds, sweep naming, W&B, SLURM params from `EBJEPA_SLURM_*` env vars.
- **Cluster utils** (added to PATH by `env.sh`): `gpus` (GPU allocation), `log -f` (tail running job stdout), `sq` (queue status), `qall` (all jobs).
- The Moving MNIST dataset requires internet access on first download. On the cluster, pre-download with `wget https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy -P $EBJEPA_DSETS/`.
