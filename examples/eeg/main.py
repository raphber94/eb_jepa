"""EEG — SSL pretraining entrypoint (self-supervised representation learning).

Research question: can two-view invariance learning on unlabeled EEG learn
features that linearly separate *normal vs abnormal* recordings, generalizing
to held-out (patient-disjoint) subjects?

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement
are marked `# TODO` below — that is the whole point of the track:
  1. the 1D encoder over [B, C=19, T]
  2. the SSL objective (two-view VICReg  *or*  predictive JEPA)
The downstream probe + metric is the third `# TODO`, in eval.py.

Run:  python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml
"""
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import Projector
from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader
from eb_jepa.losses import BCS, VICRegLoss

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP), RNNPredictor (GRU)
#   eb_jepa.losses:        VICRegLoss (inv+var+cov), VCLoss (variance+covariance)


# --------------------------------------------------------------------------- #
# 1) ENCODER  — 1D Conv stack over [B, C=19, T=2000]
# --------------------------------------------------------------------------- #
class Conv1dEncoder(nn.Module):
    """Strided Conv1d stack that downsamples time, then global-average-pools.

    Input  [B, C, T]  (e.g. [B, 19, 2000])
    Output [B, D]     via `.represent` (D = out_dim).

    Each block: Conv1d(k=7, s=2, p=3) -> BatchNorm1d -> GELU, halving T. With
    `depth` blocks the channel width ramps from `in_channels` to `out_dim` and
    T is divided by 2**depth (2000 -> 125 at depth=4).
    """

    def __init__(self, in_channels=19, out_dim=256, hidden=64, depth=4):
        super().__init__()
        # channel schedule: in -> hidden -> 2*hidden -> ... -> out_dim
        chans = [in_channels]
        for i in range(depth - 1):
            chans.append(min(hidden * (2 ** i), out_dim))
        chans.append(out_dim)
        blocks = []
        for ci, co in zip(chans[:-1], chans[1:]):
            blocks += [
                nn.Conv1d(ci, co, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(co),
                nn.GELU(),
            ]
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = out_dim

    def frames(self, x):
        """Per-timestep latent sequence [B, T', D] (before pooling).

        For the optional predictive (video-JEPA-style) framing. The two-view
        VICReg path only uses `.represent`.
        """
        h = self.backbone(x)          # [B, D, T']
        return h.transpose(1, 2)      # [B, T', D]

    def represent(self, x):
        """Global-pooled representation [B, D]."""
        h = self.backbone(x)          # [B, D, T']
        return self.pool(h).squeeze(-1)  # [B, D]

    def forward(self, x):
        return self.represent(x)


def build_encoder(cfg):
    """1D Conv encoder over an EEG window [B, C=n_channels, T] -> [B, D]."""
    return Conv1dEncoder(
        in_channels=cfg.in_channels,
        out_dim=cfg.out_dim,
        hidden=getattr(cfg, "hidden", 64),
        depth=getattr(cfg, "depth", 4),
    )


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — two-view VICReg (invariance + variance + covariance)
# --------------------------------------------------------------------------- #
class TwoViewSSL(nn.Module):
    """Two-view invariance objective: encode both augmented views, project, and
    apply VICReg (or BCS/SIGReg). The variance + covariance terms are the
    anti-collapse safeguard — watch `var_loss` from step 0."""

    def __init__(self, encoder, cfg):
        super().__init__()
        self.encoder = encoder
        proj_dim = getattr(cfg, "proj_dim", 1024)
        # spec like "256-1024-1024" : encoder D -> hidden -> proj_dim
        spec = getattr(cfg, "projector",
                       f"{encoder.out_dim}-{proj_dim}-{proj_dim}")
        self.projector = Projector(spec)
        loss_type = getattr(cfg, "loss_type", "vicreg")
        if loss_type == "vicreg":
            self.loss_fn = VICRegLoss(
                std_coeff=getattr(cfg, "std_coeff", 1.0),
                cov_coeff=getattr(cfg, "cov_coeff", 80.0),
            )
        elif loss_type == "bcs":  # SIGReg ablation
            self.loss_fn = BCS(lmbd=getattr(cfg, "lmbd", 10.0))
        else:
            raise ValueError(f"unknown loss_type: {loss_type}")

    def compute_loss(self, batch):
        v1, v2 = batch
        z1 = self.projector(self.encoder.represent(v1))
        z2 = self.projector(self.encoder.represent(v2))
        out = self.loss_fn(z1, z2)
        logs = {k: float(v.item()) for k, v in out.items()}
        return out["loss"], logs


def build_ssl(encoder, cfg):
    """Assemble the two-view SSL objective (VICReg by default)."""
    return TwoViewSSL(encoder, cfg)


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/eeg/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = EEGConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    # per-epoch metrics CSV (for the loss-curve / anti-collapse figure + monitoring)
    csv_path = os.path.join(ckpt_dir, "metrics.csv")
    csv = open(csv_path, "w")
    csv.write("epoch,loss,invariance_loss,var_loss,cov_loss\n")
    save_every = int(getattr(cfg.meta, "save_every", 0))  # 0 = only latest

    def _save(name, epoch):
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, name))

    for epoch in range(cfg.optim.epochs):
        ssl.train()
        for batch in loader:
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
        print(f"[eeg] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        csv.write("{epoch},{loss},{invariance_loss},{var_loss},{cov_loss}\n".format(
            epoch=epoch, **logs)); csv.flush()
        _save("latest.pth.tar", epoch)
        if save_every and (epoch + 1) % save_every == 0:
            _save(f"epoch_{epoch + 1}.pth.tar", epoch)
    csv.close()
    print(f"[eeg] done -> {ckpt_dir}/latest.pth.tar (metrics: {csv_path})")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/train.yaml"
    run(fname=fname)
