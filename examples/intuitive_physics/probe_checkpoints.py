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
