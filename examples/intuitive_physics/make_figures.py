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
