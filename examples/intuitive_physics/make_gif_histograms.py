"""Animated GIF histograms — normalized per-clip energy evolution over epochs.

For each clip (plausible OR impossible), computes a normalized energy:
    normed_i = E(clip_i) / mean(E(plausible clips at this epoch))

This way:
  - Plausible clips always cluster around 1  (model predicts them well)
  - Impossible clips shift to > 1 as training progresses (model gets more surprised)
  - The x-axis is fixed across epochs — easy to see the separation grow

Two overlapping histograms with high-contrast colors: blue (plausible) vs red (impossible).

Usage:
    python -m examples.intuitive_physics.make_gif_histograms \
        --ckpt_dir <path> \
        [--n_pairs 200] [--batch_size 64] [--fps 0.8] \
        [--fname examples/intuitive_physics/cfgs/eval.yaml]
"""
import io
import sys
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch

from eb_jepa.training_utils import load_checkpoint, load_config, setup_device
from examples.intuitive_physics.eval import clip_energy
from examples.intuitive_physics.main import build_jepa
from examples.intuitive_physics.stimuli import VIOLATIONS, build_probe_pairs

sns.set_theme(style="whitegrid", font_scale=1.1)

COLOR_PLAUS  = "#1565C0"   # deep blue  — plausible clips
COLOR_IMP    = "#C62828"   # deep red   — impossible clips
COLOR_REF    = "#2E7D32"   # dark green — reference line at x=1

VIOL_TITLES  = {"teleport": "Teleport", "reversal": "Reversal", "passthrough": "Passthrough"}
N_BINS       = 30


def _discover_checkpoints(ckpt_dir: Path):
    found = {}
    for p in ckpt_dir.glob("epoch_*.pth.tar"):
        try:
            ep = int(p.name.split("_")[1].split(".")[0])
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


def _optimal_threshold(normed_pla: np.ndarray, normed_imp: np.ndarray) -> tuple[float, float]:
    """Find threshold τ maximizing accuracy on the 2N unlabelled clips.
    Returns (tau, accuracy)."""
    all_e = np.concatenate([normed_pla, normed_imp])
    labels = np.r_[np.zeros(len(normed_pla)), np.ones(len(normed_imp))]
    best_acc, best_tau = 0.0, 1.0
    for tau in np.percentile(all_e, np.linspace(5, 95, 200)):
        preds = (all_e > tau).astype(float)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_tau = acc, tau
    return best_tau, best_acc


def _render_frame(normed_pla: np.ndarray, normed_imp: np.ndarray,
                  viol: str, epoch: int,
                  x_lo: float, x_hi: float, y_max: float) -> np.ndarray:
    """Render one frame with:
    - blue  = plausible clips correctly classified (E ≤ τ)
    - red   = impossible clips correctly classified (E > τ)
    - orange = false positives: plausible clips above threshold
    - purple = false negatives: impossible clips below threshold
    """
    tau, acc = _optimal_threshold(normed_pla, normed_imp)
    bins = np.linspace(x_lo, x_hi, N_BINS)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)

    # Split each distribution at threshold to color correctly/incorrectly classified
    def _hist_split(values, bins, color_ok, color_err, threshold, above_is_correct):
        for i in range(len(bins) - 1):
            mask = (values >= bins[i]) & (values < bins[i + 1])
            count = mask.sum()
            if count == 0:
                continue
            mid = (bins[i] + bins[i + 1]) / 2
            correct = (mid > threshold) == above_is_correct
            ax.bar(mid, count, width=(bins[1] - bins[0]) * 0.88,
                   color=color_ok if correct else color_err,
                   alpha=0.82, edgecolor="white", linewidth=0.3)

    # Plausible: correct if E ≤ τ (blue), error if E > τ (orange)
    _hist_split(normed_pla, bins, COLOR_PLAUS, "#FF8C00", tau, above_is_correct=False)
    # Impossible: correct if E > τ (red), error if E ≤ τ (purple)
    _hist_split(normed_imp, bins, COLOR_IMP, "#9C27B0", tau, above_is_correct=True)

    # Threshold line
    ax.axvline(tau, color="#2E7D32", linestyle="-", linewidth=2.5,
               label=f"threshold τ = {tau:.2f}  (acc={acc*100:.1f}%)")
    # Baseline
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.6,
               label="baseline (ratio = 1)")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_max * 1.22)
    ax.set_xlabel("Normalized energy  E(clip) / mean E(plausible)", fontsize=12)
    ax.set_ylabel("Number of clips", fontsize=12)
    ax.set_title(f"{VIOL_TITLES[viol]}  —  epoch {epoch}  —  unsupervised classification",
                 fontsize=12, fontweight="bold")

    fp = int((normed_pla > tau).sum())
    fn = int((normed_imp <= tau).sum())
    patch_tp = mpatches.Patch(color=COLOR_PLAUS, alpha=0.82,
                               label=f"plausible  correctly rejected  ({len(normed_pla)-fp})")
    patch_fp = mpatches.Patch(color="#FF8C00",   alpha=0.82,
                               label=f"plausible  false alarm         ({fp})")
    patch_tn = mpatches.Patch(color=COLOR_IMP,   alpha=0.82,
                               label=f"impossible correctly detected  ({len(normed_imp)-fn})")
    patch_fn = mpatches.Patch(color="#9C27B0",   alpha=0.82,
                               label=f"impossible missed              ({fn})")
    line_tau = plt.Line2D([0], [0], color="#2E7D32", linewidth=2.5,
                           label=f"optimal τ = {tau:.2f}  (acc={acc*100:.1f}%)")
    ax.legend(handles=[patch_tp, patch_fp, patch_tn, patch_fn, line_tau],
              fontsize=8.5, loc="upper right")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return imageio.v3.imread(buf)


def main():
    args = sys.argv[1:]
    if "--ckpt_dir" not in args:
        raise SystemExit(
            "usage: python -m examples.intuitive_physics.make_gif_histograms "
            "--ckpt_dir <path> [--n_pairs 200] [--batch_size 64] [--fps 0.8] "
            "[--fname examples/intuitive_physics/cfgs/eval.yaml]"
        )

    def _get(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    ckpt_dir   = Path(_get("--ckpt_dir"))
    fname      = _get("--fname", "examples/intuitive_physics/cfgs/eval.yaml")
    n_pairs    = int(_get("--n_pairs",    "200"))
    batch_size = int(_get("--batch_size", "64"))
    fps        = float(_get("--fps",      "0.8"))   # 0.8 fps → 1.25 s/frame

    cfg    = load_config(fname)
    device = setup_device(cfg.meta.get("device", "auto"))
    jepa   = build_jepa(cfg, device)

    checkpoints = _discover_checkpoints(ckpt_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {ckpt_dir}")
    print(f"Checkpoints: epochs {[e for e, _ in checkpoints]}")

    print(f"Building {n_pairs} probe pairs per violation (seed=999)…")
    pairs = build_probe_pairs(n_pairs=n_pairs, T=cfg.data.T, seed=999)

    out_dir = ckpt_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    for viol in VIOLATIONS:
        print(f"\n── {viol} ──")
        plaus  = pairs[viol]["plausible"]
        imposs = pairs[viol]["impossible"]

        # Pass 1 — collect normalized energies at every epoch to fix global axes
        all_normed_pla, all_normed_imp, epochs_done = [], [], []

        for epoch, ckpt_path in checkpoints:
            load_checkpoint(str(ckpt_path), jepa, device=device)
            jepa.eval()
            with torch.no_grad():
                e_pla = clip_energy(jepa, plaus,  cfg.model.steps, device, batch_size).numpy()
                e_imp = clip_energy(jepa, imposs, cfg.model.steps, device, batch_size).numpy()

            # Normalize by mean plausible energy → plausible clusters at 1
            baseline = e_pla.mean()
            normed_pla = e_pla / baseline
            normed_imp = e_imp / baseline

            all_normed_pla.append(normed_pla)
            all_normed_imp.append(normed_imp)
            epochs_done.append(epoch)

            print(f"  epoch {epoch:3d}  μ_plaus={normed_pla.mean():.2f}  "
                  f"μ_imp={normed_imp.mean():.2f}  "
                  f"ratio={normed_imp.mean()/normed_pla.mean():.2f}×")

        # Fixed x/y axes across all frames
        x_lo = 0.0
        x_hi = max(n.max() for n in all_normed_pla + all_normed_imp) * 1.05
        bins = np.linspace(x_lo, x_hi, N_BINS)
        y_max = int(max(
            np.histogram(n, bins=bins)[0].max()
            for n in all_normed_pla + all_normed_imp
        ))

        # Pass 2 — render and collect frames
        frames = []
        for epoch, normed_pla, normed_imp in zip(epochs_done, all_normed_pla, all_normed_imp):
            frame = _render_frame(normed_pla, normed_imp, viol, epoch,
                                  x_lo, x_hi, y_max)
            frames.append(frame)

        gif_path = out_dir / f"hist_normed_{viol}.gif"
        imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
        print(f"  ✓ {gif_path.name}  ({len(frames)} frames, {1/fps:.2f}s/frame)")

    print(f"\nAll GIFs saved to {out_dir}/")


if __name__ == "__main__":
    main()
