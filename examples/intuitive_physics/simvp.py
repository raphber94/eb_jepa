"""SimVP — Simple Video Prediction (Gao et al., CVPR 2022, arXiv:2206.05099).

Adapted for Moving MNIST 64×64 grayscale video anomaly detection.

Architecture
------------
encoder    [B*T_in, C, 64, 64]  →  [B*T_in, hid_S, 16, 16]   (frame-wise 2D CNN)
translator [B, T_in*hid_S, 16, 16] → [B, T_out*hid_S, 16, 16] (temporal mixing)
decoder    [B*T_out, hid_S, 16, 16] → [B*T_out, C, 64, 64]    (frame-wise 2D CNN)

Anomaly scoring
---------------
Train on plausible clips only (same ProceduralBouncingMNIST as EB-JEPA).
At eval time: predict frames T_in..T-1 from frames 0..T_in-1.
Per-clip anomaly score = MSE(predicted, actual) averaged over T_out frames and pixels.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvSC(nn.Module):
    """Spatial conv block: Conv2d(stride) + GroupNorm + GELU."""

    def __init__(self, C_in: int, C_out: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(1, C_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeconvSC(nn.Module):
    """Transposed conv block: ConvTranspose2d(×2) + GroupNorm + GELU."""

    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, C_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Translator(nn.Module):
    """Temporal mixing at reduced spatial resolution.

    Represents time as channel groups:
      depthwise conv  → spatial context per time step (no cross-time mixing)
      pointwise conv  → mix across time steps

    Maps [B, T_in*hid_S, H', W'] → [B, T_out*hid_S, H', W'].
    """

    def __init__(self, T_in: int, T_out: int, hid_S: int, n_layers: int = 4):
        super().__init__()
        C = T_in * hid_S
        layers = []
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(C, C, 3, padding=1, groups=T_in, bias=False),  # depthwise per t
                nn.GroupNorm(T_in, C),
                nn.GELU(),
                nn.Conv2d(C, C, 1, bias=False),                           # temporal mix
                nn.GroupNorm(T_in, C),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(C, T_out * hid_S, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x))


# ---------------------------------------------------------------------------
# SimVP model
# ---------------------------------------------------------------------------

class SimVP(nn.Module):
    """SimVP video prediction model adapted for Moving MNIST 64×64.

    Args:
        T_in:               number of input (context) frames
        T_out:              number of predicted future frames
        C:                  input channels (1 = grayscale)
        hid_S:              spatial hidden channels in encoder / decoder
        n_translator_layers: depth of the temporal translator
    """

    def __init__(
        self,
        T_in: int = 5,
        T_out: int = 5,
        C: int = 1,
        hid_S: int = 32,
        n_translator_layers: int = 4,
    ):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out

        # 64×64 → 16×16 (two stride-2 blocks)
        self.encoder = nn.Sequential(
            ConvSC(C, hid_S // 2, stride=2),
            ConvSC(hid_S // 2, hid_S, stride=2),
        )

        self.translator = Translator(T_in, T_out, hid_S, n_layers=n_translator_layers)

        # 16×16 → 64×64 (two ×2 upsamples)
        self.decoder = nn.Sequential(
            DeconvSC(hid_S, hid_S // 2),
            DeconvSC(hid_S // 2, C),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T_in, C, H, W]  context frames
        Returns:
            [B, T_out, C, H, W]    predicted future frames in [0, 1]
        """
        B, T, C, H, W = x.shape
        z = self.encoder(x.reshape(B * T, C, H, W))   # [B*T, hid_S, H', W']
        _, D, Hp, Wp = z.shape

        z = z.reshape(B, T * D, Hp, Wp)
        z = self.translator(z)                          # [B, T_out*D, H', W']
        z = z.reshape(B * self.T_out, D, Hp, Wp)

        out = self.decoder(z)                           # [B*T_out, C, H, W]
        return torch.sigmoid(out).reshape(B, self.T_out, C, H, W)

    @torch.no_grad()
    def clip_energy(
        self,
        clips: torch.Tensor,
        device: torch.device,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Per-clip pixel MSE anomaly score, shape [N].

        Input clips: [N, C, T, H, W]  (EB-JEPA storage format).
        Predicts frames T_in..T-1 from frames 0..T_in-1.
        Higher score = model is more surprised = more anomalous.
        """
        self.eval()
        all_e = []
        for start in range(0, clips.shape[0], batch_size):
            batch = clips[start : start + batch_size].to(device)  # [B, C, T, H, W]
            x = batch.permute(0, 2, 1, 3, 4)                      # [B, T, C, H, W]
            ctx = x[:, : self.T_in]                                # [B, T_in, C, H, W]
            tgt = x[:, self.T_in :]                                # [B, T_out, C, H, W]
            pred = self(ctx)                                        # [B, T_out, C, H, W]
            mse = (pred - tgt).pow(2).mean(dim=[1, 2, 3, 4])      # [B]
            all_e.append(mse.cpu())
        return torch.cat(all_e)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_simvp(cfg) -> SimVP:
    """Build SimVP from an OmegaConf config (see cfgs/simvp_train.yaml)."""
    return SimVP(
        T_in=cfg.model.T_in,
        T_out=cfg.model.T_out,
        C=cfg.model.dobs,
        hid_S=cfg.model.hid_S,
        n_translator_layers=cfg.model.n_translator_layers,
    )
