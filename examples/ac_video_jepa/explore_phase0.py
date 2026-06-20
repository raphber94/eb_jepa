"""
Phase 0 exploration harness for the "Two Rooms" AC-Video-JEPA (Track 7).

GOAL OF THIS FILE
-----------------
Let a human get "in the loop" *before* we burn any GPU hours on Dalia. It lets
you, in a few seconds on a laptop/login node (CPU is fine), answer three
questions that matter for the non-collapsing Phase 0:

    1. WHAT DOES THE ENVIRONMENT LOOK LIKE?   -> `visualize_env`
       Renders the Two Rooms world (wall + door + ball + goal), both as a still
       "task definition" image and as GIFs, so you can literally watch the ball.

    2. WHAT DOES THE MODEL SEE / IS IT WIRED RIGHT?  -> `inspect_data` / `build_model`
       Prints the exact tensor shapes of (states, actions, locations), then
       builds the *real* JEPA (same encoder/predictor/regularizer as main.py)
       and runs ONE forward pass, printing the full loss decomposition.

    3. IS IT COLLAPSING?  -> `mini_train`
       Runs a short, real training loop on freshly-generated data and prints the
       four anti-collapse signals every step, so you can SEE a healthy run vs a
       collapsing one before committing to a full Dalia job.

This is intentionally a *separate* read-only sibling of `main.py`: it imports the
same building blocks but never touches checkpoints, wandb, or SLURM. Nothing here
trains a model you keep — `mini_train` is a diagnostic, not the real run.

USAGE (uses python-fire, like main.py):
    python -m examples.ac_video_jepa.explore_phase0 visualize_env --out phase0_out
    python -m examples.ac_video_jepa.explore_phase0 inspect_data
    python -m examples.ac_video_jepa.explore_phase0 build_model
    python -m examples.ac_video_jepa.explore_phase0 mini_train --steps 150

Everything defaults to CPU and a small batch so it runs anywhere. Pass
--device cuda on a GPU box for speed.

---------------------------------------------------------------------------
WHY COLLAPSE IS THE WHOLE GAME (read this once)
---------------------------------------------------------------------------
A JEPA predicts in *latent* space, so there is a trivial cheat: make the encoder
output a constant. Then the prediction loss is 0 but the latent is information-
free and planning dies (~97% -> ~1% success). The repo prevents this with a
*force balance* of four terms (see eb_jepa/losses.py :: VC_IDM_Sim_Regularizer):

    std_loss  (HingeStd)   -> each feature must keep variance >= 1   (anti dimensional collapse)
    cov_loss  (Covariance) -> features must be decorrelated          (anti redundancy collapse)
    sim_loss_t(TemporalSim)-> consecutive frames pulled together     (this is a collapse *pressure*)
    idm_loss  (InverseDyn) -> action must be recoverable from (z_t,z_t+1)  (anti action-blind collapse)

`sim_coeff_t` actively pushes toward temporal collapse; `std`/`cov` push apart;
`idm` forces controllability info to survive. Default coeffs (cov 8 / std 16 /
sim_t 12 / idm 1) are a tuned equilibrium. Phase 0 = confirm that equilibrium
holds for our run. The helpers below surface exactly those four numbers plus two
extra diagnostics the repo does NOT log (latent effective-rank and min per-feature
std) that catch collapse earlier than the loss alone.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- Same building blocks main.py uses, imported verbatim so this harness and
#     the real training run construct *identical* models. -----------------------
from eb_jepa.architectures import ImpalaEncoder, InverseDynamicsModel, RNNPredictor
from eb_jepa.datasets.two_rooms.env import DotWall
from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig
from eb_jepa.datasets.utils import load_env_data_config
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer


# ===========================================================================
# Config + small helpers
# ===========================================================================
def _build_config(device="cpu", batch_size=16, sample_length=17, fix_wall=False):
    """Load the canonical two_rooms data config and apply small exploration overrides.

    We reuse the SAME base file the trainer reads
    (eb_jepa/datasets/two_rooms/data_config.yaml) so the world we visualize is the
    world the model will be trained on. We only shrink the batch and pin the device
    so this runs fast on CPU.

    fix_wall=False -> wall x-position and door y-position are randomized per sample
    (this is what train.yaml uses, and it is *required* for the model to generalize
    instead of memorizing one layout).
    """
    base = load_env_data_config("two_rooms")
    base.update(
        {
            "device": device,
            "batch_size": batch_size,
            "sample_length": sample_length,  # frames per trajectory the model sees
            "fix_wall": fix_wall,
        }
    )
    cfg = update_config_from_yaml(WallDatasetConfig, base)
    # The dataclass may store device as a raw string; normalize to a torch.device.
    cfg.device = torch.device(device)
    return cfg


def _make_batch(dataset, batch_size, device="cpu"):
    """Stack `batch_size` freshly-generated trajectories into a batch.

    IMPORTANT: WallDataset[i] returns ONE trajectory with shape [C, T, H, W]
    (the batch dim is squeezed; in real training the DataLoader re-batches).
    Each __getitem__ call generates a *new random* trajectory (fresh wall/door
    layout + path), so stacking N indices gives N independent samples — exactly
    what one training minibatch looks like.

    Returns (states[B,C,T,H,W], actions[B,2,T], locations[B,2,T]) as floats.
    """
    samples = [dataset[i] for i in range(batch_size)]
    states = torch.stack([s.states for s in samples]).to(device).float()
    actions = torch.stack([s.actions for s in samples]).to(device).float()
    locations = torch.stack([s.locations for s in samples]).to(device).float()
    return states, actions, locations


def _obs_to_rgb(obs):
    """Turn a 2-channel Two Rooms observation into a human-readable RGB image.

    obs: [2, H, W] tensor. Channel 0 = the ball (a Gaussian blob), channel 1 =
         the wall+border. Returns a [H, W, 3] float array in [0, 1]:
         white background, dark-gray wall, red ball.

    NOTE on scaling: the dataset frames are NORMALIZED (min-max then per-channel
    standardize), so their values are NOT in [0,255] -- the ball channel peaks
    around ~27 and the wall around ~3 (see inspect_data). The raw gym-env frames,
    by contrast, ARE uint8 0..255. To render BOTH faithfully we min-max each
    channel by its own range, which recovers full contrast regardless of scale.
    """
    obs = obs.detach().to(torch.float32).cpu().numpy()

    def _unit(a):  # min-max a single channel into [0, 1]
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    ball, wall = _unit(obs[0]), _unit(obs[1])
    # Sharpen the ball into a high-contrast marker. The dataset's normalized ball is
    # a FAINT Gaussian blob; a sub-pixel move then changes 8-bit pixels by <1/255, so
    # consecutive GIF frames quantize to byte-identical images and pillow collapses
    # them into ONE frame (a static GIF). Stretching contrast makes each frame's ball
    # a solid disk, so motion survives 8-bit quantization and the GIF actually animates.
    ball = np.clip((ball - 0.2) * 4.0, 0, 1)
    rgb = np.ones((*ball.shape, 3), dtype=np.float32)  # white canvas
    rgb *= (1.0 - wall[..., None] * 0.85)              # paint wall dark gray
    rgb[..., 1] *= (1.0 - ball)                        # remove green where ball is
    rgb[..., 2] *= (1.0 - ball)                        # remove blue  where ball is -> red
    return np.clip(rgb, 0, 1)


def _save_gif(frames_rgb, path, fps=4, upscale=6):
    """Save a list of [H,W,3] float arrays as a GIF (imageio is a repo dependency).

    upscale: nearest-neighbour zoom so the tiny 65x65 frames are actually visible in
    an IDE preview. fps is kept low so the slow ball motion is watchable.
    """
    import imageio

    frames_u8 = []
    for f in frames_rgb:
        f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        if upscale > 1:  # block-replicate pixels (keeps it crisp, no blur)
            f = np.repeat(np.repeat(f, upscale, axis=0), upscale, axis=1)
        frames_u8.append(f)
    imageio.mimsave(path, frames_u8, duration=int(1000 / fps), loop=0)
    print(f"  saved GIF  -> {path}  ({len(frames_u8)} frames, {frames_u8[0].shape[1]}x{frames_u8[0].shape[0]})")


# ===========================================================================
# 1. VISUALIZE THE ENVIRONMENT
# ===========================================================================
def visualize_env(out="phase0_out", device="cpu", seed=0, sample_length=17):
    """Render the Two Rooms world so you can SEE it before training.

    Produces, under `out/`:
      - task_definition.png : one still showing wall+door, the ball (red) at start,
                              and the goal (green) -- "from init to goal state".
      - dataset_traj.gif    : a real training trajectory (exactly the frames the
                              encoder will be fed) -- the ball moving through the door.
      - env_rollout.gif     : a live simulator rollout (env.reset + random env.step),
                              showing the actual physics + wall collisions.

    These two GIFs come from two different code paths on purpose:
      * dataset_traj  -> WallDataset (the data pipeline used in training)
      * env_rollout   -> DotWall gym env (the simulator used at planning/eval time)
    Seeing both rules out "the data looks right but the env is misconfigured" bugs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = _build_config(device=device, batch_size=4, sample_length=sample_length)

    # ---- (a) A real training trajectory, straight from the data pipeline --------
    # WallDataset[i] generates a whole BATCH of trajectories on the fly. Shapes:
    #   states:    [B, C=2, T, H, W]   (already-rendered frames: ball + wall)
    #   actions:   [B, 2, T]           (dx, dy per step)
    #   locations: [B, 2, T]           (ground-truth ball xy -- what the probe predicts)
    dataset = WallDataset(config=cfg)
    traj0 = dataset[0].states  # ONE trajectory: [C, T, H, W]
    C, T, H, W = traj0.shape
    print(f"[visualize_env] dataset trajectory: states={tuple(traj0.shape)} "
          f"(channels={C}, frames={T}, {H}x{W})")

    traj_frames = [_obs_to_rgb(traj0[:, t]) for t in range(T)]
    _save_gif(traj_frames, outdir / "dataset_traj.gif")

    # ---- (b) Task-definition still: start (red) + goal (green) + wall ------------
    env = DotWall(config=cfg, level="medium")
    obs, info = env.reset()  # obs: [2, H, W]; info has target_position + target_obs
    start_rgb = _obs_to_rgb(obs)
    goal_obs = info["target_obs"]            # [2, H, W] ball drawn at the goal
    goal_ball = goal_obs[0].to(torch.float32).cpu().numpy()
    goal_ball = goal_ball / 255.0 if goal_ball.max() > 1.5 else goal_ball
    task_img = start_rgb.copy()
    # Paint the goal blob GREEN on the same canvas as the (red) start ball.
    task_img[..., 0] *= (1.0 - goal_ball)
    task_img[..., 2] *= (1.0 - goal_ball)
    _save_png(task_img, outdir / "task_definition.png",
              title="Two Rooms: ball (red) -> goal (green)")
    print(f"[visualize_env] start={env.dot_position.tolist()}  "
          f"goal={env.target_position.tolist()}  "
          f"wall_x={float(env.wall_x):.1f}  door_y={float(env.hole_y):.1f}")

    # ---- (c) Live simulator rollout: reset + random actions ---------------------
    # This exercises the gym env exactly as planning/eval does: step() applies the
    # action, resolves wall collisions, returns the next observation.
    env2 = DotWall(config=cfg, level="medium")
    obs, info = env2.reset()
    rollout = [_obs_to_rgb(obs)]
    for _ in range(sample_length):
        action = env2.action_space.sample()              # random (dx, dy)
        obs, done, _, truncated, info = env2.step(action)  # note: env returns done in reward slot
        rollout.append(_obs_to_rgb(obs))
        if done or truncated:
            break
    _save_gif(rollout, outdir / "env_rollout.gif")

    print(f"\n[visualize_env] done -> see '{outdir}/'. Channels: 0=ball, 1=wall.\n"
          "  dataset_traj.gif = what the ENCODER is fed during training.\n"
          "  env_rollout.gif  = what the SIMULATOR produces at planning time.")


def _save_png(img_rgb, path, title=None):
    """Save a [H,W,3] image, preferring matplotlib (for a title/axis), else imageio."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.imshow(img_rgb, origin="upper")
        if title:
            plt.title(title, fontsize=9)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception:
        import imageio

        imageio.imwrite(path, (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8))
    print(f"  saved PNG  -> {path}")


# ===========================================================================
# 2a. INSPECT THE DATA
# ===========================================================================
def inspect_data(device="cpu", batch_size=8, sample_length=17, seed=0):
    """Print the exact tensors the training loop consumes, with annotations.

    The training loop iterates `for (x, a, loc, _, _) in loader` (main.py:383).
    This shows you x/a/loc shapes, dtypes and value ranges so you can confirm the
    data is sane (e.g. images in a reasonable range, actions ~unit scale) before
    feeding it to the model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = _build_config(device=device, batch_size=batch_size, sample_length=sample_length)
    dataset = WallDataset(config=cfg)
    s = dataset[0]

    def describe(name, t, meaning):
        t = t.float()
        print(f"  {name:10s} {str(tuple(t.shape)):22s} "
              f"min={t.min():7.3f} max={t.max():7.3f} mean={t.mean():7.3f}  | {meaning}")

    print("[inspect_data] ONE trajectory from WallDataset[0] "
          "(the DataLoader stacks B of these into a minibatch):")
    describe("states", s.states, "frames [C=2(ball,wall), T, H, W] -> encoder input x")
    describe("actions", s.actions, "[2, T] (dx,dy) -> predictor input a, IDM target")
    describe("locations", s.locations, "[2, T] true ball xy -> probe/aux target loc")
    describe("wall_x", s.wall_x, "[1] wall column (randomized per-sample unless fix_wall)")
    describe("door_y", s.door_y, "[1] door row")
    print("\n  Sanity checks for Phase 0:")
    print("   - states has 2 channels (ball + wall); if C!=2 the encoder is misconfigured.")
    print("   - actions are ~O(1); the predictor and IDM assume this scale.")
    print("   - each __getitem__ regenerates a fresh random wall/door+path -> model must generalize.")


# ===========================================================================
# 2b. BUILD THE REAL MODEL + ONE FORWARD PASS
# ===========================================================================
def _build_jepa(cfg_data, device, henc=32, dstc=32, dobs=2,
                cov_coeff=8, std_coeff=16, sim_coeff_t=12, idm_coeff=1):
    """Construct the JEPA EXACTLY as examples/ac_video_jepa/main.py does (lines 191-245).

    Returns (jepa, info_dict). Kept in sync with main.py on purpose: if you change
    the architecture there, change it here too, so Phase 0 diagnostics stay faithful.
    """
    img = cfg_data.img_size
    # --- encoder: Impala CNN -> a [B, F, T, h, w] latent (h=w=1 for impala) --------
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, henc, dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=dobs,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(dobs, img, img),
    ).to(device)

    # Probe the encoder output shape to size the IDM (same trick main.py uses).
    test_input = torch.rand(1, dobs, 1, img, img, device=device)
    with torch.no_grad():
        test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape

    # --- predictor: RNN that rolls the latent forward given an action --------------
    predictor = RNNPredictor(hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln).to(device)
    aencoder = nn.Identity()

    # --- inverse dynamics model: predicts the action from (z_t, z_t+1) -------------
    #     This is the LOAD-BEARING anti-collapse term -- zeroing idm_coeff collapses
    #     planning from ~97% to ~1%, so we always keep it on outside a deliberate ablation.
    idm = InverseDynamicsModel(state_dim=h * w * f, hidden_dim=256, action_dim=2).to(device)

    # --- the composite anti-collapse regularizer -----------------------------------
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cov_coeff, std_coeff=std_coeff, sim_coeff_t=sim_coeff_t,
        idm_coeff=idm_coeff, idm=idm, first_t_only=False,
        spatial_as_samples=False, idm_after_proj=False, sim_t_after_proj=False,
    ).to(device)

    jepa = JEPA(encoder, aencoder, regularizer=regularizer, predictor=predictor,
                predcost=SquareLossSeq()).to(device)

    info = {
        "latent_shape": (f, h, w),
        "encoder_params": sum(p.numel() for p in encoder.parameters()),
        "predictor_params": sum(p.numel() for p in predictor.parameters()),
        "idm_params": sum(p.numel() for p in idm.parameters()),
    }
    return jepa, info


@torch.no_grad()
def collapse_metrics(latent):
    """Compute collapse diagnostics from an encoder latent [B, C, T, h, w].

    Returns a dict. The two that the repo does NOT log but catch collapse EARLY:
      - min_feature_std : smallest per-feature std across the batch. If this slides
                          toward 0, dimensions are dying (the std hinge targets >=1).
      - effective_rank  : exp(entropy of normalized covariance eigenvalues). Equals
                          the feature count for a healthy spread; crashing toward 1
                          means everything is piling onto one direction = collapse.
    Plus temporal_var (mean sq diff between consecutive-frame latents): ~0 means
    temporal collapse (every frame maps to the same point).
    """
    B, C, T, h, w = latent.shape
    feat = latent.permute(0, 2, 3, 4, 1).reshape(-1, C).float()  # [B*T*h*w, C]
    feat_c = feat - feat.mean(0, keepdim=True)
    std = feat_c.std(0)
    # Covariance eigenvalue spectrum -> effective rank.
    cov = (feat_c.T @ feat_c) / max(1, feat_c.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    p = eig / eig.sum()
    eff_rank = torch.exp(-(p * p.log()).sum())
    # Temporal variation between consecutive frames.
    lat_t = latent.permute(2, 0, 1, 3, 4).reshape(T, B, -1).float()  # [T, B, C*h*w]
    temporal_var = (lat_t[1:] - lat_t[:-1]).pow(2).mean().item() if T > 1 else 0.0
    return {
        "mean_feature_std": std.mean().item(),
        "min_feature_std": std.min().item(),
        "effective_rank": eff_rank.item(),
        "n_features": C,
        "temporal_var": temporal_var,
    }


def build_model(device="cpu", batch_size=8, sample_length=17, seed=0):
    """Build the real JEPA and run ONE forward `unroll`, printing the loss breakdown.

    This is the fastest "is it wired right and not already degenerate at init?"
    check. It calls jepa.unroll with the SAME arguments as the training loop
    (main.py:401, autoregressive, nsteps=8) and prints every regularizer term plus
    the collapse_metrics of the encoder latent at initialization.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = _build_config(device=device, batch_size=batch_size, sample_length=sample_length)
    dataset = WallDataset(config=cfg)
    jepa, info = _build_jepa(cfg, torch.device(device))

    print("[build_model] architecture (identical to main.py):")
    print(f"   latent_shape (F,h,w) = {info['latent_shape']}")
    print(f"   params: encoder={info['encoder_params']:,}  "
          f"predictor={info['predictor_params']:,}  idm={info['idm_params']:,}")

    x, a, _ = _make_batch(dataset, batch_size, device=device)  # [B,C,T,H,W], [B,2,T]

    # Same call signature as the training loop.
    _, (loss, regl, regl_unw, regldict, pl) = jepa.unroll(
        x, a, nsteps=8, unroll_mode="autoregressive",
        ctxt_window_time=1, compute_loss=True, return_all_steps=False,
    )
    with torch.no_grad():
        latent = jepa.encoder(x)
    cm = collapse_metrics(latent)

    print("\n[build_model] ONE forward pass (untrained -> numbers are baselines, not targets):")
    print(f"   total_loss = {loss.item():.4f}   pred_loss = {pl.item():.4f}")
    print("   regularizer terms (the force balance):")
    for k, v in regldict.items():
        print(f"      {k:12s} = {v:.4f}")
    print("   latent collapse metrics:")
    for k, v in cm.items():
        print(f"      {k:16s} = {v:.4f}")
    print("\n  Reading it: at init, std_loss should be moderate (>0), effective_rank")
    print("  should be well above 1, temporal_var > 0. Training should DROP idm_loss")
    print("  and pred_loss while KEEPING effective_rank high and min_feature_std off 0.")


# ===========================================================================
# 3. MINI TRAIN: WATCH FOR COLLAPSE LIVE
# ===========================================================================
def mini_train(steps=150, device="cpu", batch_size=16, sample_length=17,
               lr=1e-3, seed=1, log_every=10,
               cov_coeff=8, std_coeff=16, sim_coeff_t=12, idm_coeff=1):
    """Run a short REAL training loop and print the four collapse signals each log step.

    This is a *diagnostic*, not the production run -- it uses a tiny batch on CPU and
    only ~`steps` iterations. Its only job is to let you watch the anti-collapse
    equilibrium hold (or fail) before launching a full job on Dalia.

    Try the healthy default, then deliberately break it to see collapse:
        python -m examples.ac_video_jepa.explore_phase0 mini_train               # healthy
        python -m examples.ac_video_jepa.explore_phase0 mini_train --idm_coeff 0 # action-blind
        python -m examples.ac_video_jepa.explore_phase0 mini_train --std_coeff 0 --cov_coeff 0  # dimensional collapse

    WHAT TO WATCH (printed each log step):
        pred      -> prediction loss; should fall.
        std/cov   -> should stay LOW; a rising std means features are dying.
        sim_t     -> low but NONZERO; ~0 = temporal collapse.
        idm       -> should fall; flat-high = latent isn't action-predictive.
        eff_rank  -> should stay >> 1; crashing toward 1 = collapse.
        min_std   -> distance of the weakest feature from death; toward 0 = danger.
    A COLLAPSE_WARNING line is printed when eff_rank or min_std cross simple thresholds.
    """
    from torch.optim import AdamW

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    cfg = _build_config(device=device, batch_size=batch_size, sample_length=sample_length)
    dataset = WallDataset(config=cfg)
    jepa, info = _build_jepa(cfg, dev, cov_coeff=cov_coeff, std_coeff=std_coeff,
                             sim_coeff_t=sim_coeff_t, idm_coeff=idm_coeff)
    n_features = info["latent_shape"][0]

    # One optimizer over the whole JEPA (encoder + predictor + idm), like main.py's
    # jepa_optimizer. We skip AMP/scheduler/grad-clip here for clarity; they don't
    # change whether the model collapses, only the speed of convergence.
    opt = AdamW(jepa.parameters(), lr=lr, weight_decay=1e-5)

    print(f"[mini_train] {steps} steps, batch={batch_size}, "
          f"coeffs(cov={cov_coeff}, std={std_coeff}, sim_t={sim_coeff_t}, idm={idm_coeff})")
    print(f"{'step':>5} {'total':>8} {'pred':>8} {'std':>7} {'cov':>7} "
          f"{'sim_t':>7} {'idm':>7} {'eff_rank':>9} {'min_std':>8}")

    jepa.train()
    for step in range(steps):
        x, a, _ = _make_batch(dataset, batch_size, device=device)  # fresh random minibatch

        opt.zero_grad()
        _, (loss, regl, regl_unw, regldict, pl) = jepa.unroll(
            x, a, nsteps=8, unroll_mode="autoregressive",
            ctxt_window_time=1, compute_loss=True, return_all_steps=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(jepa.parameters(), 2.0)  # matches grad_clip_* defaults
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                cm = collapse_metrics(jepa.encoder(x))
            print(f"{step:5d} {loss.item():8.3f} {pl.item():8.3f} "
                  f"{regldict['std_loss']:7.3f} {regldict['cov_loss']:7.3f} "
                  f"{regldict['sim_loss_t']:7.3f} {regldict['idm_loss']:7.3f} "
                  f"{cm['effective_rank']:9.2f} {cm['min_feature_std']:8.3f}")
            # Cheap, conservative collapse alarms.
            if cm["effective_rank"] < max(2.0, 0.05 * n_features):
                print("   >>> COLLAPSE_WARNING: effective_rank very low "
                      "(latent is piling onto one direction).")
            if cm["min_feature_std"] < 0.05:
                print("   >>> COLLAPSE_WARNING: a feature's std is near 0 "
                      "(dimension dying).")

    print("\n[mini_train] done. Healthy run: pred & idm fell, eff_rank stayed high, "
          "no COLLAPSE_WARNING. This is the green light to launch the full job on Dalia.")


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "visualize_env": visualize_env,
            "inspect_data": inspect_data,
            "build_model": build_model,
            "mini_train": mini_train,
        }
    )
