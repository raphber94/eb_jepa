"""Correlation-matrix heatmaps of the encoder representation, side by side.

The "identity vs blob" picture: with the covariance term ON, latent channels are
decorrelated -> a clean diagonal. With it OFF (no_cov), channels become redundant
copies of each other -> the whole grid lights up. A jury-friendly, math-free way to
show dimensional collapse (replaces the effective-rank number).

We plot the CORRELATION matrix (covariance normalized to [-1, 1], diagonal = 1) of the
raw encoder channels (dstc), measured on held-out plausible clips.

Run (defaults to full vs no_cov):
  python -m examples.intuitive_physics.make_covariance_figure \
      --ckpt $EBJEPA_CKPTS/intuitive_physics/ablation/full_seed1/latest.pth.tar \
             $EBJEPA_CKPTS/intuitive_physics/ablation/no_cov_seed1/latest.pth.tar \
      [--out figures_ablation]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from eb_jepa.training_utils import load_checkpoint, load_config, setup_device
from examples.intuitive_physics.main import build_jepa
from examples.intuitive_physics.stimuli import VIOLATIONS, build_probe_pairs


@torch.no_grad()
def _correlation_matrix(jepa, clips, device, batch_size=32):
    """[D, D] Pearson correlation of the raw encoder channels over all clip pixels."""
    jepa.eval()
    feats = []
    for start in range(0, clips.shape[0], batch_size):
        batch = clips[start:start + batch_size].to(device)
        state = jepa.encoder(batch)                                # [B, D, T, H, W]
        f = state.permute(0, 2, 3, 4, 1).reshape(-1, state.shape[1])
        feats.append(f.cpu())
    x = torch.cat(feats)                                           # [N, D]
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (x.shape[0] - 1)                            # [D, D]
    std = torch.sqrt(torch.diag(cov).clamp(min=1e-12))
    corr = cov / (std[:, None] * std[None, :])                    # normalize -> correlation
    return corr


def _label(ckpt: str) -> str:
    # .../ablation/<name>_seed1/latest.pth.tar -> <name>; the original 50-epoch run is "exp" -> "full"
    lbl = Path(ckpt).parent.name.replace("_seed1", "")
    return {"exp": "full"}.get(lbl, lbl)


def main():
    args = sys.argv[1:]

    def _get(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    out = Path(_get("--out", "figures_ablation"))
    out.mkdir(parents=True, exist_ok=True)
    fname = _get("--fname", "examples/intuitive_physics/cfgs/eval.yaml")

    if "--ckpt" in args:
        i = args.index("--ckpt") + 1
        ckpts = []
        while i < len(args) and not args[i].startswith("--"):
            ckpts.append(args[i]); i += 1
    else:
        base = Path("/lustre/work/vivatech-modelusifyoucan/rcaussign/checkpoints"
                    "/intuitive_physics/ablation")
        ckpts = [str(base / "full_seed1/latest.pth.tar"),
                 str(base / "no_cov_seed1/latest.pth.tar")]

    cfg = load_config(fname)
    device = setup_device(cfg.meta.get("device", "auto"))
    pairs = build_probe_pairs(n_pairs=cfg.probe.n_pairs, T=cfg.data.T, seed=cfg.probe.seed)
    clips = torch.cat([pairs[v]["plausible"] for v in VIOLATIONS])

    n = len(ckpts)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, ckpt in zip(axes, ckpts):
        jepa = build_jepa(cfg, device)
        load_checkpoint(ckpt, jepa, device=device)
        corr = _correlation_matrix(jepa, clips, device).numpy()
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        # off-diagonal magnitude = how redundant the channels are
        D = corr.shape[0]
        off = (abs(corr).sum() - D) / (D * (D - 1))
        ax.set_title(f"{_label(ckpt)}\nmean |off-diag corr| = {off:.2f}")
        ax.set_xlabel("latent channel"); ax.set_ylabel("latent channel")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Latent-channel correlation — diagonal = healthy, full grid = collapsed",
                 y=1.02)
    fig.tight_layout()
    path = out / "covariance.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
