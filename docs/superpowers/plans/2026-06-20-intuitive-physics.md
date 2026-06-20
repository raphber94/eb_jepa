# Intuitive Physics Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full Track 10 intuitive physics pipeline: stimulus validation → training augmentation (pixel decoder + CSV log + collapse watchdog) → per-clip energy functions → post-hoc checkpoint probe → paper-quality figure generation.

**Architecture:** Incremental, validate-first. `visualize_stimuli.py` (CPU, login node) confirms dataset correctness before any compute is spent. A 5-epoch smoke test detects collapse before committing to the 50-epoch full run. Post-hoc `probe_checkpoints.py` sweeps all saved epoch snapshots to produce a developmental trajectory. `make_figures.py` converts the two CSV logs to five publication-ready PDFs. All new code reuses existing eb_jepa bricks without modification.

**Tech Stack:** Python 3.12, PyTorch, OmegaConf, fire, matplotlib, seaborn, numpy, sklearn, imageio. All already in pyproject.toml.

## Global Constraints

- **DO NOT change** `loss.cov_coeff=100`, `loss.std_coeff=10`, `model.steps=4` — validated optimal; any change is a confound.
- Probe seed `999` must differ from training seed `1` (declared in train.yaml `meta.seed`) to avoid data leakage.
- Pixel decoder uses `lr/10` so it cannot dominate the JEPA objective.
- `clip_energy()` sanity check (`per_clip.mean() ≈ jepa.unroll(compute_loss=True)[1][4]`) must pass before any results are reported.
- Training code is **NOT run by Claude** — only the user runs it on the GPU cluster (GB200 nodes).
- Reuse existing bricks: `stimuli.py`, `vis_utils.py`, `training_utils.py`, `losses.py`, `architectures.py`, `jepa.py`, `image_decoder.py` — never rewrite them.
- Working directory: `/lustre/work/vivatech-modelusifyoucan/rberger/eb_jepa`
- Environment already activated: `source $UV_PROJECT_ENVIRONMENT/bin/activate`

---

### Task 1: Smoke-test config

**Files:**
- Create: `examples/intuitive_physics/cfgs/smoke.yaml`

**Interfaces:**
- Produces: config loadable by `load_config()` from `eb_jepa.training_utils`. Used by Task 3's modified `main.py` and Task 4's unit tests.
- Key fields: `data.n_train=500`, `data.batch_size=16`, `data.T=10`, `data.num_workers=4`, `optim.epochs=5`, `logging.save_every=1`, `logging.tqdm_silent=false`, model/loss identical to train.yaml.

- [ ] **Step 1: Create smoke.yaml**

```yaml
# Intuitive physics — fast smoke-test config for collapse detection.
# Goal: complete in ~3-5 min on one GB200 GPU.
# Identical model/loss to train.yaml; only data volume and epoch count differ.
meta:
  seed: 1
  device: auto

data:
  T: 10
  n_train: 500
  batch_size: 16
  num_workers: 4

model:
  dobs: 1
  henc: 32
  dstc: 16
  hpre: 32
  steps: 4

loss:
  std_coeff: 10.0
  cov_coeff: 100.0

optim:
  lr: 1.0e-3
  epochs: 5

logging:
  log_wandb: false
  tqdm_silent: false
  save_every: 1
```

- [ ] **Step 2: Verify it loads cleanly**

```bash
python -c "
from eb_jepa.training_utils import load_config
cfg = load_config('examples/intuitive_physics/cfgs/smoke.yaml')
assert cfg.data.n_train == 500
assert cfg.optim.epochs == 5
assert cfg.logging.save_every == 1
assert cfg.model.steps == 4
assert cfg.loss.cov_coeff == 100.0
print('smoke.yaml OK')
"
```

Expected output: `smoke.yaml OK`

---

### Task 2: Add `save_every` to train.yaml

**Files:**
- Modify: `examples/intuitive_physics/cfgs/train.yaml` (logging section only)

**Interfaces:**
- Produces: `cfg.logging.save_every == 5`, consumed by Task 3's modified `main.py`.
- `num_workers: 8` is already present; this task adds only one line.

- [ ] **Step 1: Add save_every to the logging section**

Current `logging` section in `train.yaml`:
```yaml
logging:
  log_wandb: false
  tqdm_silent: true
```

Replace with:
```yaml
logging:
  log_wandb: false
  tqdm_silent: true
  save_every: 5
```

- [ ] **Step 2: Verify**

```bash
python -c "
from eb_jepa.training_utils import load_config
cfg = load_config('examples/intuitive_physics/cfgs/train.yaml')
assert cfg.logging.save_every == 5
assert cfg.data.num_workers == 8
assert cfg.loss.cov_coeff == 100.0
print('train.yaml OK')
"
```

Expected output: `train.yaml OK`

---

### Task 3: Augment main.py (pixel decoder + CSV log + collapse watchdog + save_every)

**Files:**
- Modify: `examples/intuitive_physics/main.py`

**Interfaces:**
- Consumes: `cfg.logging.save_every` (int) from Task 2; `ImageDecoder` from `eb_jepa.image_decoder`; `JEPAProbe` from `eb_jepa.jepa`.
- Produces:
  - `{exp_dir}/train_log.csv` — columns `epoch,loss,vc_loss,pred_loss,std_loss,cov_loss,recon_loss`
  - `{exp_dir}/latest.pth.tar` — keys `model_state_dict`, `optimizer_state_dict`, `epoch`, `step`, **`decoder_state_dict`**
  - `{exp_dir}/epoch_{N}.pth.tar` — same keys, written every `save_every` epochs (epoch > 0)
  - stdout `⚠️  COLLAPSE RISK` warning when `pred_loss < 0.01 and std_loss > 0.5` after epoch 2.

- [ ] **Step 1: Rewrite main.py**

Replace the entire file with:

```python
"""Intuitive physics — video-JEPA training entrypoint.

The DATA (procedural bouncing MNIST, in ``stimuli.py``) and the TRAINING LOOP are
provided, reusing the eb_jepa core. The scientific exercise is the
violation-of-expectation PROBE in ``eval.py``: compute the per-clip latent
prediction energy (``predcost``) and compare it on matched plausible vs
impossible clips.

Run:  python -m examples.intuitive_physics.main --fname examples/intuitive_physics/cfgs/train.yaml
"""
import csv
from pathlib import Path

import fire
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import Projector, ResNet5, ResUNet, StateOnlyPredictor
from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VCLoss
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_config,
    log_config,
    log_epoch,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)
from examples.intuitive_physics.stimuli import ProceduralBouncingMNIST

logger = get_logger(__name__)

_CSV_COLUMNS = ["epoch", "loss", "vc_loss", "pred_loss", "std_loss", "cov_loss", "recon_loss"]


def _append_csv(path: Path, row: dict) -> None:
    """Append one row to a CSV log, writing the header on first call."""
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_jepa(cfg, device):
    """Video-JEPA: ResNet5 encoder + ResUNet predictor + VICReg + SquareLossSeq."""
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)
    predictor = StateOnlyPredictor(
        ResUNet(2 * cfg.model.dstc, cfg.model.hpre, cfg.model.dstc), context_length=2)
    projector = Projector(f"{cfg.model.dstc}-{cfg.model.dstc * 4}-{cfg.model.dstc * 4}")
    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    return JEPA(encoder, encoder, predictor, regularizer, SquareLossSeq(projector)).to(device)


def run(fname="examples/intuitive_physics/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    if folder is not None:
        exp_dir = Path(folder)
    elif cfg.meta.get("model_folder"):
        exp_dir = Path(cfg.meta.model_folder)
    else:
        exp_dir = get_unified_experiment_dir(
            example_name="intuitive_physics",
            sweep_name=get_default_dev_name(),
            exp_name=get_exp_name("intuitive_physics", cfg),
            seed=cfg.meta.seed,
        )
    exp_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "intuitive_physics", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_dir.name,
        tags=["intuitive_physics", f"seed_{cfg.meta.seed}"],
        enabled=cfg.logging.log_wandb,
    )

    train_set = ProceduralBouncingMNIST(split="train", n_samples=cfg.data.n_train,
                                        T=cfg.data.T, seed=cfg.meta.seed)
    loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True,
                        num_workers=cfg.data.num_workers, drop_last=True)

    jepa = build_jepa(cfg, device)

    # Pixel decoder head: ImageDecoder trained at lr/10 so it cannot dominate the JEPA objective.
    decoder = ImageDecoder(cfg.model.dstc, cfg.model.dobs, hidden_dim=cfg.model.dstc)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)

    optimizer = Adam([
        {"params": jepa.parameters(), "lr": cfg.optim.lr},
        {"params": pixel_decoder.head.parameters(), "lr": cfg.optim.lr / 10},
    ])

    save_every = cfg.logging.get("save_every", 10)
    csv_path = exp_dir / "train_log.csv"
    log_config(cfg)
    logger.info(f"Training {cfg.optim.epochs} epochs on {len(train_set)} clips -> {exp_dir}")

    global_step = 0
    for epoch in range(cfg.optim.epochs):
        jepa.train()
        pixel_decoder.train()
        last = {}
        for batch in tqdm(loader, desc=f"epoch {epoch}",
                          disable=cfg.logging.get("tqdm_silent", True)):
            x = batch["video"].to(device)
            optimizer.zero_grad()
            _, (loss, regl, _, regd, pl) = jepa.unroll(
                x, actions=None, nsteps=cfg.model.steps, unroll_mode="parallel",
                compute_loss=True)
            recon_loss = pixel_decoder(x, x)
            total_loss = loss + recon_loss
            total_loss.backward()
            optimizer.step()
            global_step += 1
            last = {"loss": loss.item(), "vc": regl.item(), "pred": pl.item(),
                    "recon": recon_loss.item(), **regd}

        # Collapse watchdog: pred_loss near 0 while std_loss still high = collapsed encoder.
        if epoch >= 2 and last.get("pred", 1.0) < 0.01 and last.get("std_loss", 0.0) > 0.5:
            logger.warning(
                f"⚠️  COLLAPSE RISK: pred_loss={last['pred']:.4f} "
                f"std_loss={last['std_loss']:.4f} — consider raising std_coeff or lowering lr"
            )

        metrics = {
            "train/loss": last["loss"], "train/vc_loss": last["vc"],
            "train/pred_loss": last["pred"], "train/std_loss": last["std_loss"],
            "train/cov_loss": last["cov_loss"], "train/recon_loss": last["recon"],
        }
        if wandb_run:
            import wandb
            wandb.log(metrics, step=global_step)

        log_epoch(epoch, {"loss": last["loss"], "vc": last["vc"],
                          "pred": last["pred"], "recon": last["recon"]},
                  total_epochs=cfg.optim.epochs)

        _append_csv(csv_path, {
            "epoch": epoch, "loss": last["loss"], "vc_loss": last["vc"],
            "pred_loss": last["pred"], "std_loss": last["std_loss"],
            "cov_loss": last["cov_loss"], "recon_loss": last["recon"],
        })

        ckpt_kwargs = dict(
            model=jepa, optimizer=optimizer, epoch=epoch, step=global_step,
            decoder_state_dict=pixel_decoder.head.state_dict(),
        )
        save_checkpoint(exp_dir / "latest.pth.tar", **ckpt_kwargs)
        if save_every > 0 and epoch > 0 and epoch % save_every == 0:
            save_checkpoint(exp_dir / f"epoch_{epoch}.pth.tar", **ckpt_kwargs)

    if wandb_run:
        import wandb
        wandb.finish()
    logger.info(f"done -> {exp_dir}/latest.pth.tar  (run probe_checkpoints.py for the energy gap)")


if __name__ == "__main__":
    fire.Fire(run)
```

- [ ] **Step 2: Smoke-import and CSV check (no GPU needed)**

```bash
python -c "
from examples.intuitive_physics.main import build_jepa, _append_csv
from eb_jepa.training_utils import load_config
import torch, tempfile, pathlib
cfg = load_config('examples/intuitive_physics/cfgs/smoke.yaml')
device = torch.device('cpu')
jepa = build_jepa(cfg, device)
# CSV helper test
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / 'log.csv'
    _append_csv(p, {'epoch':0,'loss':1.0,'vc_loss':0.5,'pred_loss':0.3,
                    'std_loss':0.2,'cov_loss':0.1,'recon_loss':0.05})
    _append_csv(p, {'epoch':1,'loss':0.9,'vc_loss':0.4,'pred_loss':0.2,
                    'std_loss':0.1,'cov_loss':0.05,'recon_loss':0.04})
    lines = p.read_text().strip().split('\n')
    assert lines[0] == 'epoch,loss,vc_loss,pred_loss,std_loss,cov_loss,recon_loss', lines[0]
    assert len(lines) == 3
print('main.py OK')
"
```

Expected output: `main.py OK`

---

### Task 4: Implement `clip_energy()` and `clip_pixel_energy()` in eval.py

**Files:**
- Modify: `examples/intuitive_physics/eval.py`
- Create: `tests/test_intuitive_physics_energy.py`

**Interfaces:**
- Consumes:
  - `jepa.encoder(clips: Tensor[B,1,T,64,64]) -> Tensor[B,D,T,H,W]`
  - `jepa.predictor(state, None)[:,:,:-1]` — advances one step, drops last output frame
  - `jepa.predictor.context_length` — int, equals 2
  - `jepa.predcost.proj` — `Projector` MLP, takes `[N, D] -> [N, D']`
- Produces:
  - `clip_energy(jepa, clips, nsteps, device, batch_size=32) -> Tensor[N]` — per-clip latent energy
  - `clip_pixel_energy(decoder_head, jepa, clips, device, batch_size=32) -> Tensor[N]` — per-clip pixel MSE
  - `_sanity_check(jepa, clips, nsteps, device, batch_size, tol)` — asserts per-clip mean ≈ batch-mean

**Key math:** `SquareLossSeq.forward()` reshapes `state [B,D,T,H,W]` via `state.transpose(0,1).flatten(1).transpose(0,1)` → `[B*T*H*W, D]`, projects, computes MSE over all elements. To get per-clip reduction, use the same reshape then split: `diff_sq.reshape(B, T*H*W*D_prime).mean(dim=1)` → `[B]`. This gives `per_clip.mean() == batch_mse` by linearity. ✓

- [ ] **Step 1: Write the failing test**

Create `tests/test_intuitive_physics_energy.py`:

```python
"""Unit tests for clip_energy() and clip_pixel_energy() in eval.py."""
import pytest
import torch

from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.training_utils import load_config
from examples.intuitive_physics.eval import (
    _sanity_check,
    clip_energy,
    clip_pixel_energy,
)
from examples.intuitive_physics.main import build_jepa


@pytest.fixture
def small_jepa():
    cfg = load_config("examples/intuitive_physics/cfgs/smoke.yaml")
    device = torch.device("cpu")
    jepa = build_jepa(cfg, device)
    jepa.eval()
    return jepa, cfg, device


def test_clip_energy_shape(small_jepa):
    jepa, cfg, device = small_jepa
    clips = torch.rand(6, 1, cfg.data.T, 64, 64)
    energies = clip_energy(jepa, clips, nsteps=cfg.model.steps, device=device, batch_size=4)
    assert energies.shape == (6,), f"Expected [6], got {energies.shape}"
    assert (energies >= 0).all(), "Energies should be non-negative"


def test_clip_energy_sanity(small_jepa):
    """Per-clip mean must match the batch-mean from jepa.unroll(compute_loss=True)."""
    jepa, cfg, device = small_jepa
    clips = torch.rand(8, 1, cfg.data.T, 64, 64)
    _sanity_check(jepa, clips, cfg.model.steps, device, batch_size=8, tol=1e-3)


def test_clip_pixel_energy_shape(small_jepa):
    jepa, cfg, device = small_jepa
    clips = torch.rand(6, 1, cfg.data.T, 64, 64)
    decoder = ImageDecoder(cfg.model.dstc, cfg.model.dobs, hidden_dim=cfg.model.dstc)
    decoder.eval()
    energies = clip_pixel_energy(decoder, jepa, clips, device=device, batch_size=4)
    assert energies.shape == (6,), f"Expected [6], got {energies.shape}"
    assert (energies >= 0).all()


def test_different_clips_give_different_energies(small_jepa):
    jepa, cfg, device = small_jepa
    clips_a = torch.rand(4, 1, cfg.data.T, 64, 64)
    clips_b = torch.rand(4, 1, cfg.data.T, 64, 64)
    e_a = clip_energy(jepa, clips_a, cfg.model.steps, device)
    e_b = clip_energy(jepa, clips_b, cfg.model.steps, device)
    assert not torch.allclose(e_a, e_b), "Different clips should give different energies"
```

- [ ] **Step 2: Run tests to confirm they fail (NotImplementedError)**

```bash
pytest tests/test_intuitive_physics_energy.py -v 2>&1 | head -20
```

Expected: `ERROR ... NotImplementedError: TODO: implement the per-clip predcost energy`

- [ ] **Step 3: Rewrite eval.py with clip_energy, clip_pixel_energy, _sanity_check**

Replace the entire file with:

```python
"""Intuitive physics — violation-of-expectation PROBE.

Loads a trained checkpoint, builds matched plausible/impossible clips, and compares
the video-JEPA's latent prediction energy (``predcost``) between them.

Run:  python -m examples.intuitive_physics.eval --ckpt <.../latest.pth.tar>
"""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.training_utils import load_checkpoint, load_config, setup_device
from examples.intuitive_physics.main import build_jepa
from examples.intuitive_physics.stimuli import VIOLATIONS, build_probe_pairs


@torch.no_grad()
def clip_energy(jepa, clips, nsteps, device, batch_size=32):
    """Per-clip latent prediction energy, shape [N].

    Replicates JEPA.unroll(parallel) but reduces per clip (mean over D,T,H,W)
    instead of batch-mean, so individual plausible/impossible clips can be compared.

    Sanity: per_clip.mean() ≈ jepa.unroll(clips, compute_loss=True)[1][4]
    """
    jepa.eval()
    all_energies = []
    N = clips.shape[0]

    for start in range(0, N, batch_size):
        batch = clips[start:start + batch_size].to(device)
        B = batch.shape[0]

        state = jepa.encoder(batch)                    # [B, D, T, H, W]
        proj = jepa.predcost.proj
        context_length = jepa.predictor.context_length  # 2

        per_clip = torch.zeros(B, device=device)
        cur_state = state

        for _ in range(nsteps):
            pred = jepa.predictor(cur_state, None)[:, :, :-1]            # [B, D, T-1, H, W]
            cur_state = torch.cat(
                [state[:, :, :context_length], pred], dim=2
            )                                                              # [B, D, T, H, W]

            # Reshape [B, D, T, H, W] -> [B*T*H*W, D] matching SquareLossSeq exactly
            B_, D_, T_, H_, W_ = state.shape
            s_flat = state.transpose(0, 1).flatten(1).transpose(0, 1)     # [B*T*H*W, D]
            p_flat = cur_state.transpose(0, 1).flatten(1).transpose(0, 1) # [B*T*H*W, D]
            s_proj = proj(s_flat)   # [B*T*H*W, D']
            p_proj = proj(p_flat)   # [B*T*H*W, D']

            D_prime = s_proj.shape[-1]
            diff_sq = (s_proj - p_proj).pow(2)                            # [B*T*H*W, D']
            # Reshape to [B, T*H*W*D'] and mean per clip (rows are B-major)
            per_clip += diff_sq.reshape(B_, T_ * H_ * W_ * D_prime).mean(dim=1)

        per_clip /= nsteps
        all_energies.append(per_clip.cpu())

    return torch.cat(all_energies)  # [N]


@torch.no_grad()
def clip_pixel_energy(decoder_head, jepa, clips, device, batch_size=32):
    """Per-clip pixel reconstruction MSE, shape [N].

    Passes encoder representations through the pixel decoder head and measures
    reconstruction error. Baseline: spatial structure only, no temporal prediction.
    """
    jepa.eval()
    decoder_head.eval()
    all_energies = []
    N = clips.shape[0]

    for start in range(0, N, batch_size):
        batch = clips[start:start + batch_size].to(device)
        enc = jepa.encoder(batch)              # [B, D, T, H, W]
        recon = decoder_head(enc)              # [B, 1, T, H, W]
        mse = ((recon - batch) ** 2).mean(dim=[1, 2, 3, 4])  # [B]
        all_energies.append(mse.cpu())

    return torch.cat(all_energies)  # [N]


def _sanity_check(jepa, clips, nsteps, device, batch_size=32, tol=1e-3):
    """Assert per_clip.mean() ≈ jepa.unroll(compute_loss=True)[1][4] (batch-mean predcost)."""
    sample = clips[:batch_size].to(device)
    per_clip = clip_energy(jepa, sample, nsteps, device, batch_size=batch_size)
    jepa.eval()
    with torch.no_grad():
        _, losses = jepa.unroll(sample, actions=None, nsteps=nsteps,
                                unroll_mode="parallel", compute_loss=True)
    batch_mean = losses[4].item()
    delta = abs(per_clip.mean().item() - batch_mean)
    assert delta < tol, (
        f"clip_energy sanity check FAILED: per_clip.mean()={per_clip.mean():.6f} "
        f"vs jepa.unroll predcost={batch_mean:.6f} (delta={delta:.2e} > tol={tol})"
    )


def _auroc(e_pla, e_imp):
    y = np.r_[np.zeros(len(e_pla)), np.ones(len(e_imp))]
    return float(roc_auc_score(y, np.r_[e_pla.numpy(), e_imp.numpy()]))


def main():
    if "--ckpt" not in sys.argv:
        raise SystemExit("usage: python -m examples.intuitive_physics.eval --ckpt <path> "
                         "[--fname examples/intuitive_physics/cfgs/eval.yaml]")
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    fname = (sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv
             else "examples/intuitive_physics/cfgs/eval.yaml")
    cfg = load_config(fname)
    device = setup_device(cfg.meta.get("device", "auto"))

    jepa = build_jepa(cfg, device)
    load_checkpoint(ckpt, jepa, device=device)
    jepa.eval()

    # Load pixel decoder if the checkpoint was produced by the augmented main.py
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    decoder_head = None
    if "decoder_state_dict" in raw:
        decoder = ImageDecoder(cfg.model.dstc, cfg.model.dobs, hidden_dim=cfg.model.dstc)
        decoder.load_state_dict(raw["decoder_state_dict"])
        decoder = decoder.to(device)
        decoder.eval()
        decoder_head = decoder

    pairs = build_probe_pairs(n_pairs=cfg.probe.n_pairs, T=cfg.data.T, seed=cfg.probe.seed)
    nsteps = cfg.model.steps

    # Sanity check on first violation's plausible clips
    _sanity_check(jepa, pairs[VIOLATIONS[0]]["plausible"], nsteps, device)
    print("✓ clip_energy sanity check passed")

    header = f"{'violation':12s} {'lat_gap':>10s} {'lat_auroc':>9s}"
    if decoder_head is not None:
        header += f" {'pix_gap':>10s} {'pix_auroc':>9s}"
    print(header)

    results = {}
    for v in VIOLATIONS:
        e_pla = clip_energy(jepa, pairs[v]["plausible"], nsteps, device)
        e_imp = clip_energy(jepa, pairs[v]["impossible"], nsteps, device)
        lat_gap = float(e_imp.mean() - e_pla.mean())
        lat_auroc = _auroc(e_pla, e_imp)
        results[v] = {"lat_gap": lat_gap, "lat_auroc": lat_auroc}

        line = f"{v:12s} {lat_gap:>10.3e} {lat_auroc:>9.3f}"
        if decoder_head is not None:
            px_pla = clip_pixel_energy(decoder_head, jepa, pairs[v]["plausible"], device)
            px_imp = clip_pixel_energy(decoder_head, jepa, pairs[v]["impossible"], device)
            pix_gap = float(px_imp.mean() - px_pla.mean())
            pix_auroc = _auroc(px_pla, px_imp)
            results[v].update({"pix_gap": pix_gap, "pix_auroc": pix_auroc})
            line += f" {pix_gap:>10.3e} {pix_auroc:>9.3f}"
        print(line)

    return results


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect them all to pass**

```bash
pytest tests/test_intuitive_physics_energy.py -v
```

Expected output:
```
tests/test_intuitive_physics_energy.py::test_clip_energy_shape PASSED
tests/test_intuitive_physics_energy.py::test_clip_energy_sanity PASSED
tests/test_intuitive_physics_energy.py::test_clip_pixel_energy_shape PASSED
tests/test_intuitive_physics_energy.py::test_different_clips_give_different_energies PASSED
4 passed in ...s
```

---

### Task 5: visualize_stimuli.py (standalone, CPU, login node)

**Files:**
- Create: `examples/intuitive_physics/visualize_stimuli.py`

**Interfaces:**
- Consumes: `build_probe_pairs(n_pairs, T, seed, violations)` from `stimuli.py`; `prepare_frame`, `add_border`, `save_gif_as_pdf_unroll` from `eb_jepa.vis_utils`; `imageio` (already in pyproject).
- Produces (all under `$EBJEPA_CKPTS/intuitive_physics/stimuli_viz/`):
  - `strip_{violation}_pair{i}.pdf` — 2-row PDF (plausible top, impossible+red borders bottom), one per pair × violation
  - `compare_{violation}.gif` — side-by-side animated GIF (plausible left | impossible right), one per violation
  - `trajectory_{violation}.pdf` — digit centroid x/y vs frame for pair 0, one per violation
  - stdout summary table: `violation, n_pairs, t_v min/max/mean`

- [ ] **Step 1: Create visualize_stimuli.py**

```python
"""Standalone stimuli validation — no training or GPU required.

Generates visual sanity checks for the matched plausible/impossible pairs.
Run this BEFORE submitting any cluster job.

Usage: python -m examples.intuitive_physics.visualize_stimuli
"""
import os
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from eb_jepa.vis_utils import add_border, prepare_frame, save_gif_as_pdf_unroll
from examples.intuitive_physics.stimuli import VIOLATIONS, build_probe_pairs

N_PAIRS = 4
T = 10
SEED = 42
CKPTS_ROOT = os.environ.get("EBJEPA_CKPTS", "checkpoints")
OUT_DIR = Path(CKPTS_ROOT) / "intuitive_physics" / "stimuli_viz"


def _centroid(frame_np: np.ndarray):
    """Return (cx, cy) pixel centroid of a [H, W] float frame."""
    total = float(frame_np.sum())
    if total < 1e-6:
        return float("nan"), float("nan")
    h, w = frame_np.shape
    cx = float((frame_np * np.arange(w, dtype=np.float32)[None, :]).sum() / total)
    cy = float((frame_np * np.arange(h, dtype=np.float32)[:, None]).sum() / total)
    return cx, cy


def _strip_pdf(plaus: torch.Tensor, imposs: torch.Tensor, t_v: int,
               violation: str, out_path: Path) -> None:
    """2-row PDF strip: plausible top, impossible bottom (red border at/after t_v).

    plaus, imposs: [1, T, 64, 64] in [0, 1].
    """
    T_clip = plaus.shape[1]
    plaus_frames, imposs_frames = [], []
    for t in range(T_clip):
        pf = prepare_frame(plaus[0, t])   # [64, 64, 3] uint8
        inf = prepare_frame(imposs[0, t])
        if t >= t_v:
            inf = add_border(inf, color=(255, 0, 0), width=3)
        plaus_frames.append(pf)
        imposs_frames.append(inf)

    save_gif_as_pdf_unroll(
        [plaus_frames, imposs_frames],
        str(out_path),
        row_labels=["plausible", f"impossible ({violation})"],
        title=f"Violation: {violation}  |  t_v={t_v}",
        figsize_per_frame=(0.9, 0.9),
    )


def _compare_gif(plaus: torch.Tensor, imposs: torch.Tensor, t_v: int,
                 out_path: Path) -> None:
    """Side-by-side GIF: plausible left | impossible right, 5 fps."""
    T_clip = plaus.shape[1]
    frames = []
    for t in range(T_clip):
        pf = prepare_frame(plaus[0, t])
        inf = prepare_frame(imposs[0, t])
        if t >= t_v:
            inf = add_border(inf, color=(255, 0, 0), width=3)
        frames.append(np.concatenate([pf, inf], axis=1))  # [64, 128, 3]
    imageio.mimsave(str(out_path), frames, fps=5, loop=0)


def _trajectory_pdf(plaus: torch.Tensor, imposs: torch.Tensor,
                    t_v: int, violation: str, out_path: Path) -> None:
    """Digit centroid x/y vs frame for one pair."""
    T_clip = plaus.shape[1]
    cx_p, cy_p, cx_i, cy_i = [], [], [], []
    for t in range(T_clip):
        px, py = _centroid(plaus[0, t].numpy())
        ix, iy = _centroid(imposs[0, t].numpy())
        cx_p.append(px); cy_p.append(py)
        cx_i.append(ix); cy_i.append(iy)

    frames = list(range(T_clip))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), dpi=150)
    for ax, yp, yi, label in [
        (ax1, cx_p, cx_i, "centroid x (px)"),
        (ax2, cy_p, cy_i, "centroid y (px)"),
    ]:
        ax.plot(frames, yp, "b-o", label="plausible", markersize=4)
        ax.plot(frames, yi, "r-o", label=f"impossible ({violation})", markersize=4)
        ax.axvline(t_v, color="gray", linestyle="--", alpha=0.7, label=f"t_v={t_v}")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xticks(frames)
    ax2.set_xlabel("frame", fontsize=9)
    fig.suptitle(f"Digit trajectory — {violation}", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    pairs = build_probe_pairs(n_pairs=N_PAIRS, T=T, seed=SEED, violations=VIOLATIONS)

    print(f"\n{'violation':12s} {'n_pairs':>8s} {'t_v min':>8s} {'t_v max':>8s} {'t_v mean':>9s}")
    print("-" * 52)

    for viol in VIOLATIONS:
        plaus  = pairs[viol]["plausible"]   # [N, 1, T, 64, 64]
        imposs = pairs[viol]["impossible"]  # [N, 1, T, 64, 64]
        tvs    = pairs[viol]["t_v"]         # [N]
        tv_f   = tvs.float()

        print(f"{viol:12s} {N_PAIRS:>8d} {tv_f.min().item():>8.0f} "
              f"{tv_f.max().item():>8.0f} {tv_f.mean().item():>9.1f}")

        for i in range(N_PAIRS):
            _strip_pdf(plaus[i], imposs[i], int(tvs[i].item()), viol,
                       OUT_DIR / f"strip_{viol}_pair{i}.pdf")

        _compare_gif(plaus[0], imposs[0], int(tvs[0].item()),
                     OUT_DIR / f"compare_{viol}.gif")
        _trajectory_pdf(plaus[0], imposs[0], int(tvs[0].item()), viol,
                        OUT_DIR / f"trajectory_{viol}.pdf")

    n_strips = len(VIOLATIONS) * N_PAIRS
    n_traj   = len(VIOLATIONS)
    n_gifs   = len(VIOLATIONS)
    print(f"\nSaved {n_strips} strip PDFs + {n_traj} trajectory PDFs + {n_gifs} comparison GIFs")
    print(f"→ {OUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (login node, no GPU, ~5-10 seconds)**

```bash
python -m examples.intuitive_physics.visualize_stimuli
```

Expected stdout:
```
Output directory: <EBJEPA_CKPTS>/intuitive_physics/stimuli_viz

violation     n_pairs   t_v min   t_v max  t_v mean
----------------------------------------------------
teleport            4         3         7       5.0
reversal            4         3         7       5.0
passthrough         4         3         7       5.0

Saved 12 strip PDFs + 3 trajectory PDFs + 3 comparison GIFs
→ <EBJEPA_CKPTS>/intuitive_physics/stimuli_viz
```

(t_v values depend on seed=42 simulation; exact numbers may differ but range 3-7 is expected for T=10.)

Visually verify: plausible row shows smooth bouncing; impossible row shows a clear discontinuity at the first red-bordered frame.

---

### Task 6: probe_checkpoints.py (post-hoc sweep, CPU or GPU)

**Files:**
- Create: `examples/intuitive_physics/probe_checkpoints.py`

**Interfaces:**
- Consumes: `clip_energy(jepa, clips, nsteps, device, batch_size) -> Tensor[N]` and `clip_pixel_energy(decoder_head, jepa, clips, device, batch_size) -> Tensor[N]` from Task 4; `_auroc(e_pla, e_imp) -> float` from eval.py; `build_jepa(cfg, device)` from Task 3; `build_probe_pairs(n_pairs, T, seed)` from stimuli.py; checkpoint files with keys `model_state_dict`, `decoder_state_dict`, `epoch`.
- Produces: `{ckpt_dir}/probe_results.csv` — columns `epoch,violation,latent_gap,pixel_gap,latent_auroc,pixel_auroc,e_plaus_mean,e_imp_mean,px_plaus_mean,px_imp_mean`; live stdout table.

- [ ] **Step 1: Create probe_checkpoints.py**

```python
"""Post-hoc VoE probe — sweeps epoch_*.pth.tar checkpoints and appends to probe_results.csv.

For each checkpoint found in ckpt_dir, loads jepa + pixel decoder, runs clip_energy()
and clip_pixel_energy() on held-out matched pairs, and records the energy gap and AUROC
for each violation type.

Usage:
    python -m examples.intuitive_physics.probe_checkpoints \
        --ckpt_dir $EBJEPA_CKPTS/intuitive_physics/<sweep>/<exp>_seed1 \
        [--fname examples/intuitive_physics/cfgs/eval.yaml] \
        [--n_pairs 200] [--batch_size 32]
"""
import csv
import sys
from pathlib import Path

import torch

from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.training_utils import load_checkpoint, load_config, setup_device
from examples.intuitive_physics.eval import _auroc, clip_energy, clip_pixel_energy
from examples.intuitive_physics.main import build_jepa
from examples.intuitive_physics.stimuli import VIOLATIONS, build_probe_pairs

_CSV_COLUMNS = [
    "epoch", "violation",
    "latent_gap", "pixel_gap",
    "latent_auroc", "pixel_auroc",
    "e_plaus_mean", "e_imp_mean",
    "px_plaus_mean", "px_imp_mean",
]


def _discover_checkpoints(ckpt_dir: Path):
    """Return sorted (epoch, path) list. Includes latest.pth.tar if no epoch_N file covers it."""
    found = {}
    for p in ckpt_dir.glob("epoch_*.pth.tar"):
        try:
            ep = int(p.stem.split("_")[1])
            found[ep] = p
        except (IndexError, ValueError):
            continue
    latest = ckpt_dir / "latest.pth.tar"
    if latest.exists():
        raw = torch.load(latest, map_location="cpu", weights_only=False)
        ep = raw.get("epoch", -1)
        if ep not in found:
            found[ep] = latest
    return sorted(found.items())


def _probe_one(ckpt_path: Path, jepa, cfg, pairs, device, batch_size):
    """Load checkpoint, run probe for all violations, return list of row dicts."""
    load_checkpoint(str(ckpt_path), jepa, device=device)
    jepa.eval()

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = raw.get("epoch", -1)

    decoder_head = None
    if "decoder_state_dict" in raw:
        decoder_head = ImageDecoder(
            cfg.model.dstc, cfg.model.dobs, hidden_dim=cfg.model.dstc
        ).to(device)
        decoder_head.load_state_dict(raw["decoder_state_dict"])
        decoder_head.eval()

    rows = []
    for viol in VIOLATIONS:
        plaus  = pairs[viol]["plausible"]
        imposs = pairs[viol]["impossible"]

        e_pla = clip_energy(jepa, plaus,  cfg.model.steps, device, batch_size)
        e_imp = clip_energy(jepa, imposs, cfg.model.steps, device, batch_size)
        lat_gap   = float(e_imp.mean() - e_pla.mean())
        lat_auroc = _auroc(e_pla, e_imp)

        pix_gap = pix_auroc = px_pla_mean = px_imp_mean = float("nan")
        if decoder_head is not None:
            px_pla = clip_pixel_energy(decoder_head, jepa, plaus,  device, batch_size)
            px_imp = clip_pixel_energy(decoder_head, jepa, imposs, device, batch_size)
            pix_gap   = float(px_imp.mean() - px_pla.mean())
            pix_auroc = _auroc(px_pla, px_imp)
            px_pla_mean = float(px_pla.mean())
            px_imp_mean = float(px_imp.mean())

        rows.append({
            "epoch": epoch, "violation": viol,
            "latent_gap": lat_gap, "pixel_gap": pix_gap,
            "latent_auroc": lat_auroc, "pixel_auroc": pix_auroc,
            "e_plaus_mean": float(e_pla.mean()), "e_imp_mean": float(e_imp.mean()),
            "px_plaus_mean": px_pla_mean, "px_imp_mean": px_imp_mean,
        })
    return rows


def main():
    args = sys.argv[1:]
    if "--ckpt_dir" not in args:
        raise SystemExit(
            "usage: python -m examples.intuitive_physics.probe_checkpoints "
            "--ckpt_dir <path> [--fname cfgs/eval.yaml] [--n_pairs 200] [--batch_size 32]"
        )

    def _get(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    ckpt_dir   = Path(_get("--ckpt_dir"))
    fname      = _get("--fname", "examples/intuitive_physics/cfgs/eval.yaml")
    n_pairs    = int(_get("--n_pairs", "200"))
    batch_size = int(_get("--batch_size", "32"))

    cfg    = load_config(fname)
    device = setup_device(cfg.meta.get("device", "auto"))
    jepa   = build_jepa(cfg, device)

    print(f"Building {n_pairs} probe pairs per violation (seed=999, held-out from training)…")
    pairs = build_probe_pairs(n_pairs=n_pairs, T=cfg.data.T, seed=999)

    checkpoints = _discover_checkpoints(ckpt_dir)
    if not checkpoints:
        raise SystemExit(f"No epoch_*.pth.tar or latest.pth.tar found in {ckpt_dir}")

    csv_path = ckpt_dir / "probe_results.csv"
    write_header = not csv_path.exists()

    print(f"\n{'epoch':>6s} {'violation':>12s} {'lat_gap':>10s} {'lat_auc':>8s} "
          f"{'pix_gap':>10s} {'pix_auc':>8s}")
    print("-" * 60)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for epoch, ckpt_path in checkpoints:
            rows = _probe_one(ckpt_path, jepa, cfg, pairs, device, batch_size)
            for row in rows:
                writer.writerow(row)
                f.flush()
                print(
                    f"{row['epoch']:>6d} {row['violation']:>12s} "
                    f"{row['latent_gap']:>10.3e} {row['latent_auroc']:>8.3f} "
                    f"{row['pixel_gap']:>10.3e} {row['pixel_auroc']:>8.3f}"
                )

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-import check (no checkpoint needed)**

```bash
python -c "
from examples.intuitive_physics.probe_checkpoints import _discover_checkpoints, _probe_one
print('probe_checkpoints.py imports OK')
"
```

Expected: `probe_checkpoints.py imports OK`

---

### Task 7: make_figures.py (five paper-quality PDFs)

**Files:**
- Create: `examples/intuitive_physics/make_figures.py`

**Interfaces:**
- Consumes:
  - `{ckpt_dir}/train_log.csv` — columns `epoch,loss,vc_loss,pred_loss,std_loss,cov_loss,recon_loss`
  - `{ckpt_dir}/probe_results.csv` — columns `epoch,violation,latent_gap,pixel_gap,latent_auroc,pixel_auroc,e_plaus_mean,e_imp_mean,px_plaus_mean,px_imp_mean`
  - `{ckpt_dir}/latest.pth.tar` — final checkpoint for Fig 3 distribution and Fig 4 timeline
  - `clip_energy(jepa, clips, nsteps, device) -> Tensor[N]` from Task 4
- Produces (all in `{ckpt_dir}/figures/`):
  - `fig1_energy_gap.pdf` — latent gap vs epoch, one line per violation
  - `fig2_latent_vs_pixel.pdf` — latent gap (solid) vs pixel gap (dashed) per violation
  - `fig3_distributions_{violation}.pdf` — overlapping energy histograms at final epoch (×3)
  - `fig4_surprise_timeline.pdf` — 1-step prediction energy per frame for one teleport pair
  - `fig5_training_health.pdf` — pred+recon loss (top panel) + std+cov loss (bottom panel) vs epoch

- [ ] **Step 1: Create make_figures.py**

```python
"""Generate paper-quality figures from training logs and probe results.

Usage:
    python -m examples.intuitive_physics.make_figures \
        --ckpt_dir $EBJEPA_CKPTS/intuitive_physics/<sweep>/<exp>_seed1 \
        [--fname examples/intuitive_physics/cfgs/eval.yaml]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from eb_jepa.training_utils import load_checkpoint, load_config, setup_device
from examples.intuitive_physics.eval import clip_energy
from examples.intuitive_physics.main import build_jepa
from examples.intuitive_physics.stimuli import VIOLATIONS, build_probe_pairs

sns.set_theme(style="whitegrid", font_scale=1.0)
VIOL_COLORS = {"teleport": "#e41a1c", "reversal": "#377eb8", "passthrough": "#4daf4a"}
DPI = 300


def _load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found at {path}. "
            "Run main.py (for train_log) or probe_checkpoints.py (for probe_results) first."
        )
    return pd.read_csv(path)


def fig1_energy_gap(probe: pd.DataFrame, out: Path) -> None:
    """Latent energy gap (impossible − plausible) vs epoch, one line per violation."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
    for viol in VIOLATIONS:
        df = probe[probe["violation"] == viol].sort_values("epoch")
        ax.plot(df["epoch"], df["latent_gap"], label=viol,
                color=VIOL_COLORS[viol], marker="o", markersize=4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="chance (gap=0)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy gap  E(impossible) − E(plausible)")
    ax.set_title("Latent VoE signal vs training — Track 10")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig2_latent_vs_pixel(probe: pd.DataFrame, out: Path) -> None:
    """Latent gap (solid) vs pixel gap (dashed) — the A/B test from Garrido et al."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
    for viol in VIOLATIONS:
        df = probe[probe["violation"] == viol].sort_values("epoch")
        ax.plot(df["epoch"], df["latent_gap"], label=f"{viol} (latent)",
                color=VIOL_COLORS[viol], marker="o", markersize=3)
        ax.plot(df["epoch"], df["pixel_gap"], label=f"{viol} (pixel)",
                color=VIOL_COLORS[viol], linestyle="--", marker="s", markersize=3)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy gap")
    ax.set_title("Latent vs pixel prediction — VoE gap")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig3_distributions(probe: pd.DataFrame, out_dir: Path, cfg, jepa, device) -> None:
    """Overlapping histograms of per-clip energies at final checkpoint (200 pairs each)."""
    final_epoch = int(probe["epoch"].max())
    pairs = build_probe_pairs(n_pairs=200, T=cfg.data.T, seed=999)

    for viol in VIOLATIONS:
        e_pla = clip_energy(jepa, pairs[viol]["plausible"],  cfg.model.steps, device)
        e_imp = clip_energy(jepa, pairs[viol]["impossible"], cfg.model.steps, device)

        lo = min(e_pla.min(), e_imp.min()).item()
        hi = max(e_pla.max(), e_imp.max()).item()
        bins = np.linspace(lo, hi, 30)

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=DPI)
        ax.hist(e_pla.numpy(), bins=bins, alpha=0.6, color="#2166ac", label="plausible")
        ax.hist(e_imp.numpy(), bins=bins, alpha=0.6, color="#d6604d",
                label=f"impossible ({viol})")
        ax.set_xlabel("Latent prediction energy")
        ax.set_ylabel("Count")
        ax.set_title(f"{viol}  (epoch {final_epoch})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = out_dir / f"fig3_distributions_{viol}.pdf"
        plt.savefig(str(out), bbox_inches="tight")
        plt.close()
        print(f"  ✓ {out.name}")


def fig4_surprise_timeline(cfg, jepa, device, out: Path) -> None:
    """Per-frame 1-step prediction energy for one matched teleport pair.

    For each consecutive pair of frames (t, t+1), feeds state[:,:,t:t+2] to the
    predictor and computes the projected MSE between predicted frame t+2 and actual
    frame t+2. A spike at/after t_v on the impossible clip is the VoE signal.
    """
    pairs = build_probe_pairs(n_pairs=1, T=cfg.data.T, seed=999)
    viol  = "teleport"
    plaus  = pairs[viol]["plausible"]    # [1, 1, T, 64, 64]
    imposs = pairs[viol]["impossible"]   # [1, 1, T, 64, 64]
    t_v    = int(pairs[viol]["t_v"][0].item())
    T_clip = cfg.data.T
    proj   = jepa.predcost.proj

    def _frame_energies(clip_tensor):
        clip_dev = clip_tensor.to(device)
        state = jepa.encoder(clip_dev)  # [1, D, T, H, W]
        energies = []
        for t in range(T_clip - 2):
            ctx    = state[:, :, t:t+2]        # [1, D, 2, H, W]
            pred   = jepa.predictor(ctx, None) # [1, D, 1, H, W]
            target = state[:, :, t+2:t+3]      # [1, D, 1, H, W]
            B_, D_, T_, H_, W_ = pred.shape
            p_flat = pred.transpose(0,1).flatten(1).transpose(0,1)
            t_flat = target.transpose(0,1).flatten(1).transpose(0,1)
            p_proj = proj(p_flat)
            t_proj = proj(t_flat)
            energies.append((p_proj - t_proj).pow(2).mean().item())
        return energies

    with torch.no_grad():
        plaus_e  = _frame_energies(plaus)
        imposs_e = _frame_energies(imposs)

    predicted_frames = list(range(2, T_clip))  # frame indices of the predicted target
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=DPI)
    ax.plot(predicted_frames, plaus_e,  "b-o", markersize=5, label="plausible")
    ax.plot(predicted_frames, imposs_e, "r-o", markersize=5, label="impossible (teleport)")
    ax.axvline(t_v, color="gray", linestyle="--", label=f"t_v={t_v}")
    ax.set_xlabel("Predicted frame index")
    ax.set_ylabel("1-step prediction energy")
    ax.set_title("Per-frame surprise — teleport pair")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig5_training_health(train_log: pd.DataFrame, out: Path) -> None:
    """Two-panel: pred+recon loss (top) and std+cov loss (bottom) vs epoch."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=DPI, sharex=True)
    ax1.plot(train_log["epoch"], train_log["pred_loss"],  label="pred_loss",  color="#e41a1c")
    ax1.plot(train_log["epoch"], train_log["recon_loss"], label="recon_loss", color="#ff7f00")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.set_title("Training health")
    ax2.plot(train_log["epoch"], train_log["std_loss"], label="std_loss",  color="#377eb8")
    ax2.plot(train_log["epoch"], train_log["cov_loss"], label="cov_loss",  color="#984ea3")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Regularization loss")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def main():
    args = sys.argv[1:]
    if "--ckpt_dir" not in args:
        raise SystemExit(
            "usage: python -m examples.intuitive_physics.make_figures "
            "--ckpt_dir <path> [--fname cfgs/eval.yaml]"
        )

    def _get(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    ckpt_dir = Path(_get("--ckpt_dir"))
    fname    = _get("--fname", "examples/intuitive_physics/cfgs/eval.yaml")

    cfg    = load_config(fname)
    device = setup_device(cfg.meta.get("device", "auto"))

    train_log = _load_csv(ckpt_dir / "train_log.csv",      "train_log.csv")
    probe     = _load_csv(ckpt_dir / "probe_results.csv",  "probe_results.csv")

    jepa = build_jepa(cfg, device)
    load_checkpoint(str(ckpt_dir / "latest.pth.tar"), jepa, device=device)
    jepa.eval()

    out_dir = ckpt_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    print(f"Writing figures to {out_dir}/")

    fig1_energy_gap(probe, out_dir / "fig1_energy_gap.pdf")
    fig2_latent_vs_pixel(probe, out_dir / "fig2_latent_vs_pixel.pdf")
    fig3_distributions(probe, out_dir, cfg, jepa, device)
    fig4_surprise_timeline(cfg, jepa, device, out_dir / "fig4_surprise_timeline.pdf")
    fig5_training_health(train_log, out_dir / "fig5_training_health.pdf")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-import check and mock fig5 test**

```bash
python -c "
from examples.intuitive_physics.make_figures import fig5_training_health
import pandas as pd, pathlib, tempfile
train_log = pd.DataFrame({
    'epoch':      list(range(5)),
    'pred_loss':  [1.0, 0.8, 0.6, 0.4, 0.2],
    'recon_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
    'std_loss':   [0.9, 0.7, 0.5, 0.3, 0.1],
    'cov_loss':   [0.4, 0.3, 0.2, 0.1, 0.05],
})
with tempfile.TemporaryDirectory() as d:
    fig5_training_health(train_log, pathlib.Path(d) / 'test.pdf')
from examples.intuitive_physics.make_figures import fig1_energy_gap
import numpy as np
probe = pd.DataFrame({
    'epoch':      [0,0,0,5,5,5],
    'violation':  ['teleport','reversal','passthrough'] * 2,
    'latent_gap': [0.1, -0.05, 0.08, 0.5, 0.3, 0.4],
    'pixel_gap':  [0.02, 0.01, 0.03, 0.05, 0.02, 0.04],
})
with tempfile.TemporaryDirectory() as d:
    fig1_energy_gap(probe, pathlib.Path(d) / 'fig1.pdf')
print('make_figures.py OK')
"
```

Expected: `  ✓ test.pdf` then `  ✓ fig1.pdf` then `make_figures.py OK`

---

## Execution Order (user runs Steps A and B on the GPU cluster)

```bash
# Step 0 — validate stimuli (login node, ~10 sec, no GPU)
python -m examples.intuitive_physics.visualize_stimuli
# Inspect: strip PDFs show smooth plausible + discontinuous impossible at red border

# Step A — smoke test (user runs on cluster, ~5 min)
python -m examples.intuitive_physics.main \
    --fname examples/intuitive_physics/cfgs/smoke.yaml
# Watch for ⚠️  COLLAPSE RISK in stdout. If seen: stop, raise std_coeff in smoke.yaml.

# Step B — full training (user runs on cluster, ~2 h)
python -m examples.intuitive_physics.main \
    --fname examples/intuitive_physics/cfgs/train.yaml
# Or via SLURM: python -m examples.launch_sbatch --example intuitive_physics --single \
#     --fname examples/intuitive_physics/cfgs/train.yaml

# Step C — post-hoc probe sweep (login node or CPU, ~10 min)
python -m examples.intuitive_physics.probe_checkpoints \
    --ckpt_dir $EBJEPA_CKPTS/intuitive_physics/<sweep_name>/<exp_name>_seed1

# Step D — generate figures (login node)
python -m examples.intuitive_physics.make_figures \
    --ckpt_dir $EBJEPA_CKPTS/intuitive_physics/<sweep_name>/<exp_name>_seed1
```
