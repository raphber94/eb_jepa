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
