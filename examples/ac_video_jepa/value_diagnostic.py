"""
Diagnose whether a trained GoalValueHead is INFORMATIVE or COLLAPSED.

A useful value must DECREASE with distance-to-goal: V(state, goal) high when the
state is near the goal, low when far. If V is ~constant (collapsed) the planner's
cost 1-V is flat -> no signal -> near-random planning (our Track 7 symptom).

We test in TWO regimes:
  - sample_length=17  : the TRAINING regime (near goals, <=16 steps)
  - sample_length=64  : the PLANNING-like regime (far goals)
If V tracks distance at 17 but not at 64, the problem is OUT-OF-DISTRIBUTION far
goals (the fix: train on longer windows). If it fails even at 17, it's a full collapse.

Usage:
    python -m examples.ac_video_jepa.value_diagnostic \
        --ckpt evidence/track7_75937/train/model/e-11.pth.tar --sample_length 17
"""

import numpy as np
import torch

from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset
from eb_jepa.state_decoder import GoalValueHead
from examples.ac_video_jepa.explore_phase0 import _build_config, _build_jepa, _make_batch


def diagnose(ckpt, device="cuda", batch_size=64, sample_length=17):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg = _build_config(device=str(dev), batch_size=batch_size, sample_length=sample_length)
    dataset = WallDataset(config=cfg)
    jepa, info = _build_jepa(cfg, dev)
    value_head = GoalValueHead(info["latent_shape"][0]).to(dev)

    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}
    jepa.load_state_dict(sd, strict=False)
    value_head.load_state_dict(ck["value_head_state_dict"])
    jepa.eval()
    value_head.eval()

    x, a, loc = _make_batch(dataset, batch_size, device=str(dev))
    with torch.no_grad():
        z = jepa.encoder(x).float()       # [B, 512, T, 1, 1]
        goal = z[:, :, -1:]               # endpoint latent
        T = z.shape[2]
        V, D = [], []
        for t in range(T - 1):
            v = value_head(z[:, :, t:t + 1], goal).reshape(-1)              # [B]
            d = (loc[:, :, t] - loc[:, :, -1]).pow(2).sum(1).sqrt()         # [B] dist-to-goal
            V.append(v.cpu().numpy())
            D.append(d.cpu().numpy())
    V = np.concatenate(V)
    D = np.concatenate(D)
    corr = float(np.corrcoef(V, D)[0, 1])
    med = np.median(D)

    print(f"\n===== VALUE DIAGNOSTIC  (sample_length={sample_length}) =====")
    print(f"  ckpt: {ckpt}")
    print(f"  V:  mean={V.mean():.4f}  std={V.std():.4f}  range=[{V.min():.3f}, {V.max():.3f}]")
    print(f"  corr(V, distance-to-goal) = {corr:+.3f}   (want strongly NEGATIVE)")
    print(f"  V near goal (d<median) = {V[D < med].mean():.4f}")
    print(f"  V far  goal (d>=median)= {V[D >= med].mean():.4f}")
    verdict = ("INFORMATIVE" if corr < -0.3 and V.std() > 0.02
               else "COLLAPSED/UNINFORMATIVE")
    print(f"  VERDICT: {verdict}")
    print("=" * 52)
    return {"corr": corr, "std": float(V.std()), "verdict": verdict}


if __name__ == "__main__":
    import fire

    fire.Fire(diagnose)
