"""SimVP training entrypoint for Moving MNIST intuitive physics.

Mirrors examples/intuitive_physics/main.py structure so both models
are trained under identical conditions (same dataset, same budget, same seed).

Run:
    python -m examples.intuitive_physics.train_simvp \
        --fname examples/intuitive_physics/cfgs/simvp_train.yaml
"""

import csv
from pathlib import Path

import fire
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_config,
    log_config,
    setup_device,
    setup_seed,
    setup_wandb,
)
from examples.intuitive_physics.simvp import SimVP, build_simvp
from examples.intuitive_physics.stimuli import ProceduralBouncingMNIST

logger = get_logger(__name__)

_CSV_COLUMNS = ["epoch", "train_loss"]


def _append_csv(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run(fname="examples/intuitive_physics/cfgs/simvp_train.yaml", cfg=None,
        folder=None, **overrides):
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
            example_name="intuitive_physics_simvp",
            sweep_name=get_default_dev_name(),
            exp_name=get_exp_name("simvp", cfg),
            seed=cfg.meta.seed,
        )
    exp_dir.mkdir(parents=True, exist_ok=True)

    setup_wandb(
        project="eb_jepa",
        config={"example": "simvp", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_dir.name,
        tags=["simvp", f"seed_{cfg.meta.seed}"],
        enabled=cfg.logging.log_wandb,
    )

    train_set = ProceduralBouncingMNIST(
        split="train", n_samples=cfg.data.n_train, T=cfg.data.T, seed=cfg.meta.seed
    )
    loader = DataLoader(
        train_set, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, drop_last=True,
    )

    model = build_simvp(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"SimVP parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=cfg.optim.lr)
    criterion = nn.MSELoss()

    T_in = cfg.model.T_in
    save_every = cfg.logging.get("save_every", 10)
    csv_path = exp_dir / "train_log.csv"
    log_config(cfg)
    logger.info(f"Training {cfg.optim.epochs} epochs on {len(train_set)} clips → {exp_dir}")

    global_step = 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=f"epoch {epoch}",
                          disable=cfg.logging.get("tqdm_silent", True)):
            # batch["video"]: [B, C, T, H, W] with C=1
            x = batch["video"].to(device).permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            ctx = x[:, :T_in]     # [B, T_in, C, H, W]
            tgt = x[:, T_in:]     # [B, T_out, C, H, W]

            optimizer.zero_grad()
            pred = model(ctx)     # [B, T_out, C, H, W]
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"epoch {epoch:3d}  loss={avg_loss:.5f}")
        _append_csv(csv_path, {"epoch": epoch, "train_loss": avg_loss})

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(ckpt, exp_dir / "latest.pth.tar")
        if save_every > 0 and epoch > 0 and epoch % save_every == 0:
            torch.save(ckpt, exp_dir / f"epoch_{epoch}.pth.tar")

    logger.info(f"done → {exp_dir}/latest.pth.tar")


if __name__ == "__main__":
    fire.Fire(run)
