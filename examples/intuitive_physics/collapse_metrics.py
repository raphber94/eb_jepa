"""Collapse diagnostics for a trained intuitive-physics video-JEPA.

These metrics are computed on the *encoder representation* directly and do NOT depend
on the training loss, so they detect collapse even when pred_loss looks healthy:

- feat_std_mean   : mean real std per latent channel. ~1+ healthy, ~0 collapsed.
                    (Note: train_log's `std_loss` is the hinge penalty relu(1-std),
                     here we report the actual std.)
- feat_std_min    : worst channel — exposes partial collapse.
- eff_rank        : effective rank = exp(entropy of normalized covariance eigenvalues),
                    out of dstc channels. ~dstc healthy, ~1 dimensional collapse.
- eff_rank_ratio  : eff_rank / dstc in [0, 1].

Run:  python -m examples.intuitive_physics.collapse_metrics --ckpt <.../latest.pth.tar>
      python -m examples.intuitive_physics.collapse_metrics --ckpt A.pth.tar B.pth.tar
"""
import sys

import torch

from eb_jepa.training_utils import load_checkpoint, load_config, setup_device
from examples.intuitive_physics.main import build_jepa
from examples.intuitive_physics.stimuli import VIOLATIONS, build_probe_pairs


@torch.no_grad()
def _gather_features(jepa, clips, device, batch_size=32):
    """Encode clips and flatten to [N_samples, dstc] (channels = feature dim)."""
    feats = []
    for start in range(0, clips.shape[0], batch_size):
        batch = clips[start:start + batch_size].to(device)
        state = jepa.encoder(batch)                       # [B, D, T, H, W]
        f = state.permute(0, 2, 3, 4, 1).reshape(-1, state.shape[1])  # [B*T*H*W, D]
        feats.append(f.cpu())
    return torch.cat(feats)                               # [N, D]


def _effective_rank(cov):
    """exp(Shannon entropy of normalized eigenvalues) — a soft, continuous rank."""
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = eig / (eig.sum() + 1e-12)
    p = p[p > 0]
    entropy = -(p * p.log()).sum()
    return float(entropy.exp())


@torch.no_grad()
def collapse_metrics(jepa, clips, device):
    """Return collapse diagnostics dict for the encoder representation on `clips`."""
    jepa.eval()
    feats = _gather_features(jepa, clips, device)         # [N, D]
    feats_c = feats - feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0)                                # [D]
    cov = (feats_c.T @ feats_c) / (feats.shape[0] - 1)    # [D, D]
    D = feats.shape[1]
    eff = _effective_rank(cov)
    return {
        "feat_std_mean": float(std.mean()),
        "feat_std_min": float(std.min()),
        "feat_std_max": float(std.max()),
        "eff_rank": eff,
        "dstc": D,
        "eff_rank_ratio": eff / D,
    }


def _run_one(ckpt, cfg, device, clips):
    jepa = build_jepa(cfg, device)
    load_checkpoint(ckpt, jepa, device=device)
    return collapse_metrics(jepa, clips, device)


def main():
    if "--ckpt" not in sys.argv:
        raise SystemExit("usage: python -m examples.intuitive_physics.collapse_metrics "
                         "--ckpt <path> [<path> ...] [--fname cfgs/eval.yaml]")
    i = sys.argv.index("--ckpt") + 1
    ckpts = []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        ckpts.append(sys.argv[i]); i += 1
    fname = (sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv
             else "examples/intuitive_physics/cfgs/eval.yaml")

    cfg = load_config(fname)
    device = setup_device(cfg.meta.get("device", "auto"))

    # Held-out plausible clips, pooled across violation types, as the probe set.
    pairs = build_probe_pairs(n_pairs=cfg.probe.n_pairs, T=cfg.data.T, seed=cfg.probe.seed)
    clips = torch.cat([pairs[v]["plausible"] for v in VIOLATIONS])

    print(f"{'checkpoint':45s} {'std_mean':>9s} {'std_min':>9s} "
          f"{'eff_rank':>9s} {'rank_ratio':>11s}")
    for ckpt in ckpts:
        m = _run_one(ckpt, cfg, device, clips)
        name = ckpt if len(ckpt) <= 45 else "..." + ckpt[-42:]
        print(f"{name:45s} {m['feat_std_mean']:>9.4f} {m['feat_std_min']:>9.4f} "
              f"{m['eff_rank']:>9.3f} {m['eff_rank_ratio']:>11.3f}  (/{m['dstc']})")


if __name__ == "__main__":
    main()
