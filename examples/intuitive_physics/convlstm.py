"""ConvLSTM — Convolutional LSTM for Moving MNIST video prediction.

Reference: Shi et al., "Convolutional LSTM Network: A Machine Learning Approach
           for Precipitation Nowcasting", NeurIPS 2015.

The original Moving MNIST benchmark paper. Used here as a pixel-prediction
anomaly detector: train on plausible clips, score violations by prediction MSE.

Architecture
------------
- Stack of ConvLSTMCells processing frames sequentially
- Context: first T_in frames  → hidden state
- Prediction: T_out future frames generated autoregressively
- Output conv: hidden → pixel frame in [0,1]

Operates at full 64×64 resolution (unlike SimVP which downsamples).
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell computing all 4 gates with one convolution."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ):
        """
        Args:
            x: [B, C_in, H, W]
            h: [B, hid, H, W]  previous hidden state
            c: [B, hid, H, W]  previous cell state
        Returns:
            h_new, c_new: [B, hid, H, W]
        """
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, B: int, H: int, W: int, device: torch.device):
        z = torch.zeros(B, self.hidden_channels, H, W, device=device)
        return z, z.clone()


class ConvLSTM(nn.Module):
    """ConvLSTM video predictor for Moving MNIST 64×64.

    Args:
        T_in:            number of input (context) frames
        T_out:           number of predicted future frames
        C:               input channels (1 = grayscale)
        hidden_channels: hidden state channels per LSTM layer
        n_layers:        number of stacked ConvLSTM layers
    """

    def __init__(
        self,
        T_in: int = 5,
        T_out: int = 5,
        C: int = 1,
        hidden_channels: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out

        cells = []
        for i in range(n_layers):
            in_c = C if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(in_c, hidden_channels))
        self.cells = nn.ModuleList(cells)

        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, C, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T_in, C, H, W]  context frames
        Returns:
            [B, T_out, C, H, W]    predicted future frames in [0, 1]
        """
        B, T, C, H, W = x.shape
        states = [cell.init_hidden(B, H, W, x.device) for cell in self.cells]

        # Encode context: run T_in frames through the LSTM stack
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                h, c = states[i]
                h, c = cell(inp, h, c)
                states[i] = (h, c)
                inp = h

        # Predict autoregressively: feed each prediction as next input
        preds = []
        inp = x[:, -1]  # last context frame as seed
        for _ in range(self.T_out):
            for i, cell in enumerate(self.cells):
                h, c = states[i]
                h, c = cell(inp, h, c)
                states[i] = (h, c)
                inp = h
            pred = torch.sigmoid(self.head(inp))    # [B, C, H, W]
            preds.append(pred)
            inp = pred                               # autoregressive feed

        return torch.stack(preds, dim=1)            # [B, T_out, C, H, W]

    @torch.no_grad()
    def clip_energy(
        self,
        clips: torch.Tensor,
        device: torch.device,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Per-clip pixel MSE anomaly score, shape [N].

        Same interface as SimVP.clip_energy — higher = more anomalous.
        """
        self.eval()
        all_e = []
        for start in range(0, clips.shape[0], batch_size):
            batch = clips[start : start + batch_size].to(device)   # [B, C, T, H, W]
            x = batch.permute(0, 2, 1, 3, 4)                       # [B, T, C, H, W]
            ctx = x[:, : self.T_in]
            tgt = x[:, self.T_in :]
            pred = self(ctx)
            mse = (pred - tgt).pow(2).mean(dim=[1, 2, 3, 4])       # [B]
            all_e.append(mse.cpu())
        return torch.cat(all_e)


def build_convlstm(cfg) -> ConvLSTM:
    return ConvLSTM(
        T_in=cfg.model.T_in,
        T_out=cfg.model.T_out,
        C=cfg.model.dobs,
        hidden_channels=cfg.model.hidden_channels,
        n_layers=cfg.model.n_layers,
    )
