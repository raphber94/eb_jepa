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
