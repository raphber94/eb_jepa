# Intuitive Physics Probe — Design Spec
**Date:** 2026-06-20  
**Track:** VivaTech Hackathon — Track 10 "Does intuitive physics emerge?"  
**Scope:** End-to-end pipeline from stimuli validation to paper-quality figures.

---

## 1. Context & Scientific Rationale

Garrido et al. (2025) showed that V-JEPA — trained to predict future frames in **latent representation space** — develops a violation-of-expectation (VoE) signal: its prediction energy is reliably higher on physically impossible videos than on matched plausible ones (98% pairwise accuracy on IntPhys, zero-shot). The central contrast is **latent prediction vs. pixel prediction**: VideoMAEv2 (pixel-space) stays near chance.

This pipeline replicates that result in miniature on Moving MNIST with the EB-JEPA library:
- Train a video-JEPA (ResNet5 encoder + ResUNet predictor + VICReg) on plausible single-digit bouncing clips.
- Probe whether the latent `predcost` energy gap (impossible − plausible) emerges over training.
- Compare against a pixel-decoder baseline trained jointly, testing whether the latent signal is strictly more discriminative.

### Violation types (3, fixed)
| Name | Physics broken | How |
|---|---|---|
| `teleport` | Continuity | Position jumps ≥18px at frame `t_v` |
| `reversal` | Inertia | Velocity instantly negates in free flight |
| `passthrough` | Impenetrability | Digit wraps through wall (modular arithmetic) |

Pairs share frames `0..t_v-1` exactly — low-level bias control is built-in by construction.

---

## 2. Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| `examples/intuitive_physics/stimuli.py` | **Unchanged** | Already correct; 3 violation types |
| `examples/intuitive_physics/cfgs/smoke.yaml` | **New** | Fast collapse-detection config (5 epochs) |
| `examples/intuitive_physics/cfgs/train.yaml` | **Modified** | Add `save_every: 5` for checkpoint sweep |
| `examples/intuitive_physics/main.py` | **Modified** | Add pixel decoder + CSV log + collapse watchdog |
| `examples/intuitive_physics/eval.py` | **Modified** | Implement `clip_energy()` + `clip_pixel_energy()` |
| `examples/intuitive_physics/visualize_stimuli.py` | **New** | Standalone dataset validator, no training needed |
| `examples/intuitive_physics/probe_checkpoints.py` | **New** | Post-hoc checkpoint sweep → `probe_results.csv` |
| `examples/intuitive_physics/make_figures.py` | **New** | CSV → 5 paper-quality PDF figures |

---

## 3. Dataset & Stimuli

`stimuli.py` is used as-is. `visualize_stimuli.py` is a standalone validation script:

**Inputs:** none (uses `build_probe_pairs(n_pairs=4, seed=42)` with MNIST test digits).  
**Outputs** to `$EBJEPA_CKPTS/intuitive_physics/stimuli_viz/`:
- One PDF strip per violation type (2 rows × 4 columns: plausible top, impossible bottom; frames at/after `t_v` red-bordered) using `vis_utils.save_gif_as_pdf_unroll()` + `add_border()`.
- One side-by-side GIF per violation type (plausible left, impossible right) using `vis_utils.save_gif()`.
- Trajectory plots (digit center x/y vs frame) for 1 pair per type, confirming divergence at `t_v`.

**Runtime:** ~5 seconds on login node (CPU only). Run this before any cluster job.

---

## 4. Training Architecture

### 4a. Smoke-test config (`cfgs/smoke.yaml`)
```yaml
data:
  n_train: 500
  batch_size: 16
optim:
  epochs: 5
logging:
  save_every: 1
  tqdm_silent: false
  log_wandb: false
```
All model/loss hyperparameters identical to `train.yaml`. Goal: ~3–5 min on one GB200. Collapse verdict readable from stdout.

### 4b. `main.py` modifications (additive only)

**Pixel decoder** (identical pattern to `video_jepa/main.py`):
```python
decoder = ImageDecoder(cfg.model.dstc, cfg.model.dobs)
pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)
# Optimizer: pixel_decoder.head at lr/10
recon_loss = pixel_decoder(x, x)
total_loss = jepa_loss + recon_loss
```

**Combined checkpoint saving** — both `jepa` and `pixel_decoder.head` saved in the same `.pth.tar`:
```python
torch.save({
    "model_state_dict": jepa.state_dict(),
    "decoder_state_dict": pixel_decoder.head.state_dict(),
    "epoch": epoch, "step": global_step,
}, path)
```

**Per-epoch CSV log** appended to `{exp_dir}/train_log.csv`:
```
epoch, loss, vc_loss, pred_loss, std_loss, cov_loss, recon_loss
```

**Collapse watchdog** — after each epoch:
```python
if pred_loss < 0.01 and std_loss > 0.5:
    logger.warning("⚠️  COLLAPSE RISK: pred_loss={:.4f} std_loss={:.4f} "
                   "— consider raising std_coeff or lowering lr")
```

### 4c. `train.yaml` modification
Add `save_every: 5` to `logging` section so checkpoints are saved at epochs 5, 10, 15 … 50.

---

## 5. `clip_energy()` and Pixel Baseline

### `clip_energy(jepa, clips, nsteps, device, batch_size=32) → Tensor[N]`

Manual parallel-mode unroll with per-clip reduction (bypasses `JEPA.unroll()` to avoid forced batch-mean):

```python
state = jepa.encoder(clips)                        # [B, D, T, H, W]
proj = jepa.predcost.proj                          # the shared Projector MLP
context_length = jepa.predictor.context_length     # = 2
ploss = torch.zeros(B)
for _ in range(nsteps):
    pred = jepa.predictor(state, None)[:, :, :-1]
    pred = torch.cat([state[:, :, :context_length], pred], dim=2)
    # Project both and compute per-clip MSE
    s_proj = proj(state.transpose(0,1).flatten(1).transpose(0,1))
    p_proj = proj(pred.transpose(0,1).flatten(1).transpose(0,1))
    diff = F.mse_loss(s_proj, p_proj, reduction='none')   # [B*T*H*W, D']
    ploss += diff.reshape(B, -1).mean(dim=1)
ploss /= nsteps
```

**Sanity check** (asserted at startup): `abs(ploss.mean() - jepa.unroll(..., compute_loss=True)[1][4].item()) < 1e-4`.

Processes clips in mini-batches of `batch_size=32` to avoid OOM on 400-clip probe set.

### `clip_pixel_energy(pixel_decoder, jepa, clips, device, batch_size=32) → Tensor[N]`

```python
recon = pixel_decoder.head(jepa.encoder(clips))   # [B, 1, T, 64, 64]
mse = ((recon - clips) ** 2).mean(dim=[1,2,3,4])  # [B]
```

Note: this is reconstruction error (encoder → decoder), not temporal prediction error. It measures whether the spatial structure is captured, not the temporal dynamics — the distinction that the paper claims matters.

---

## 6. Post-hoc Probe (`probe_checkpoints.py`)

```bash
python -m examples.intuitive_physics.probe_checkpoints \
    --ckpt_dir $EBJEPA_CKPTS/intuitive_physics/dev_.../resnet_..._seed1 \
    --fname examples/intuitive_physics/cfgs/eval.yaml
```

**Steps:**
1. Glob `epoch_*.pth.tar` + `latest.pth.tar`, sort by epoch number.
2. For each checkpoint: load `jepa` + `pixel_decoder.head`, build probe pairs once (cached), run `clip_energy()` and `clip_pixel_energy()` for all 3 violation types × 200 pairs.
3. Append to `{ckpt_dir}/probe_results.csv`:
   ```
   epoch, violation, latent_gap, pixel_gap, latent_auroc, pixel_auroc,
   e_plaus_mean, e_imp_mean, px_plaus_mean, px_imp_mean
   ```
4. Print live table row per checkpoint so progress is visible.

Probe pairs built once with `build_probe_pairs(n_pairs=200, seed=999)` (seed held out from training generator, which uses `seed=2025`).

---

## 7. Figure Generation (`make_figures.py`)

```bash
python -m examples.intuitive_physics.make_figures \
    --ckpt_dir $EBJEPA_CKPTS/intuitive_physics/dev_.../resnet_..._seed1
```

Reads `train_log.csv` + `probe_results.csv` from `{ckpt_dir}`, writes to `{ckpt_dir}/figures/`:

| Figure | File | Content |
|---|---|---|
| 1 | `fig1_energy_gap.pdf` | Latent energy gap (impossible − plausible) vs epoch, 3 lines (one per violation). Dashed baseline at 0. |
| 2 | `fig2_latent_vs_pixel.pdf` | Same axes, solid=latent gap, dashed=pixel gap. The A/B test. |
| 3 | `fig3_distributions_{v}.pdf` | Overlapping histograms of per-clip energies, plausible=blue / impossible=red, at final checkpoint. One PDF per violation type. |
| 4 | `fig4_surprise_timeline.pdf` | Per-frame energy for one matched pair, with vertical line at `t_v`. Uses single-step unroll sliding window. |
| 5 | `fig5_training_health.pdf` | Two-panel: `pred_loss` + `recon_loss` (top), `std_loss` + `cov_loss` (bottom) vs epoch. |

All figures use `matplotlib` + `seaborn`, saved at 300 DPI.

---

## 8. Execution Order

```
# Step 0 — validate stimuli (login node, no GPU)
python -m examples.intuitive_physics.visualize_stimuli

# Step 1 — smoke test (single GPU, ~5 min)
python -m examples.intuitive_physics.main \
    --fname examples/intuitive_physics/cfgs/smoke.yaml
# → inspect stdout for ⚠️ COLLAPSE RISK warnings and pred_loss trajectory

# Step 2 — full training (single GPU, ~2h)
python -m examples.intuitive_physics.main \
    --fname examples/intuitive_physics/cfgs/train.yaml
# or via SLURM:
python -m examples.launch_sbatch --example intuitive_physics --single \
    --fname examples/intuitive_physics/cfgs/train.yaml

# Step 3 — post-hoc probe sweep (login node or CPU job, ~10 min)
python -m examples.intuitive_physics.probe_checkpoints \
    --ckpt_dir $EBJEPA_CKPTS/intuitive_physics/<sweep_name>/<exp_name>_seed1

# Step 4 — generate figures
python -m examples.intuitive_physics.make_figures \
    --ckpt_dir $EBJEPA_CKPTS/intuitive_physics/<sweep_name>/<exp_name>_seed1
```

---

## 9. Expected Outcomes & Interpretation

| Outcome | Interpretation |
|---|---|
| `latent_gap > 0` for all 3 violations at epoch 50 | VoE signal emerged ✓ |
| `latent_gap > pixel_gap` | Latent prediction is more discriminative ✓ (paper's claim) |
| `latent_gap ≈ 0` at epoch 0, grows over training | Signal is learned, not architectural bias ✓ |
| `std_loss` stays high, `pred_loss` → 0 | Encoder collapsed — raise `std_coeff`, lower LR |
| `latent_gap ≈ pixel_gap` | Either both work or neither does; worth investigating clip-level distribution |
| `reversal` gap < `teleport`/`passthrough` gaps | Expected — reversal is the hardest violation (magnitude-matched, no OOD position) |

---

## 10. Key Invariants

- **Do not change** `loss.cov_coeff=100`, `loss.std_coeff=10`, `model.steps=4`.
- Probe seed `999` must remain different from training seed `2025` (no data leakage).
- Pixel decoder trained with `lr/10` to prevent it from dominating the JEPA objective.
- `clip_energy()` sanity check must pass before any probe results are reported.
