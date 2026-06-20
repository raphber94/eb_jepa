"""Inference-only latent visualizations for the intuitive-physics video-JEPA.

Uses an ALREADY-TRAINED checkpoint (no training). Produces three deliverables:

  1. fig_latent_position_map.pdf  — PCA + t-SNE of the globally pooled per-frame
     latent, colored by the digit's true (x, y). A clean colour gradient means the
     holistic representation encodes position (the "Two Rooms colored by XY" probe).
     Also prints a linear-probe R^2 for position decoded from the pooled latent.

  2. fig_rollout_breakdown.pdf     — feed the predictor its own output and roll the
     latent forward many steps. The latent-drift curve vs the true future shows WHEN
     prediction quality breaks down. fig_rollout_filmstrip.pdf decodes those latents
     so you can see true vs reconstructed vs predicted frames.

  3. fig_latent_vs_pixel.pdf       — why latent-space evaluation beats checking after
     decoding: pixel-reconstruction error stays flat across the violation frame while
     latent prediction energy spikes; a per-clip scatter shows latent separates
     plausible/impossible where pixel error overlaps.

Run:
    python -m examples.intuitive_physics.visualize_latent \
        --ckpt /lustre/work/vivatech-modelusifyoucan/shared/dev_2026-06-20_04-03/exp_seed1/latest.pth.tar \
        [--out <dir>] [--fname examples/intuitive_physics/cfgs/eval.yaml]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection="3d")
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
    KNeighborsClassifier,
    LocalOutlierFactor,
    NearestNeighbors,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.training_utils import load_checkpoint, load_config, setup_device
from examples.intuitive_physics.eval import clip_energy, clip_pixel_energy
from examples.intuitive_physics.main import build_jepa
from examples.intuitive_physics.stimuli import (
    CANVAS,
    DIGIT,
    VIOLATIONS,
    build_probe_pairs,
    load_mnist_digits,
    render_clip,
    sample_plausible,
)

sns.set_theme(style="whitegrid", context="talk")
DPI = 200


def save_fig(fig, out):
    """Save a figure as both PDF (vector, for the paper) and PNG (for slides)."""
    fig.savefig(str(out), bbox_inches="tight")
    fig.savefig(str(Path(out).with_suffix(".png")), bbox_inches="tight", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def gen_plausible(n, T, seed, split="test"):
    """`n` plausible clips plus the digit-center (row, col) per frame.

    Returns (videos [n, 1, T, 64, 64], centers [n, T, 2]).
    """
    digits = load_mnist_digits(split)
    rng = np.random.RandomState(seed)
    vids, centers = [], []
    for _ in range(n):
        d = digits[rng.randint(len(digits))]
        pos = sample_plausible(rng, T)          # [T, 2] top-left
        vids.append(render_clip(pos, d))
        centers.append(pos + DIGIT / 2.0)        # [T, 2] center (row, col)
    return torch.from_numpy(np.stack(vids)), np.stack(centers)


@torch.no_grad()
def pooled_latents(jepa, videos, device, batch_size=32):
    """Spatially mean-pooled per-frame latent: [n, T, D]."""
    jepa.eval()
    out = []
    for s in range(0, videos.shape[0], batch_size):
        state = jepa.encoder(videos[s:s + batch_size].to(device))  # [B, D, T, H, W]
        out.append(state.mean(dim=[3, 4]).transpose(1, 2).cpu())   # [B, T, D]
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------- #
# Deliverable 1 — latent position map
# --------------------------------------------------------------------------- #
def fig_latent_position_map(jepa, device, out, n_clips=80, T=10, seed=4321):
    videos, centers = gen_plausible(n_clips, T, seed)
    emb = pooled_latents(jepa, videos, device)            # [n, T, D]
    X = emb.reshape(-1, emb.shape[-1])                     # [n*T, D]
    cy = centers[:, :, 0].reshape(-1)                      # row  (y)
    cx = centers[:, :, 1].reshape(-1)                      # col  (x)

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    pca = PCA(n_components=2).fit_transform(Xs)
    perp = max(5, min(30, (len(Xs) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                learning_rate="auto", random_state=0).fit_transform(Xs)

    # Linear probe: how well does the pooled latent linearly decode position?
    r2 = LinearRegression().fit(X, np.c_[cy, cx]).score(X, np.c_[cy, cx])

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=DPI)
    for row, (emb2d, name) in enumerate([(pca, "PCA"), (tsne, "t-SNE")]):
        for col, (c, cname) in enumerate([(cx, "x (col)"), (cy, "y (row)")]):
            ax = axes[row, col]
            sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=c, cmap="viridis", s=10)
            ax.set_title(f"{name}  —  colored by digit {cname}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.suptitle(
        f"Pooled latent vs digit position  (linear-probe R^2 = {r2:.3f})\n"
        "clean colour gradients ⇒ the representation encodes position",
        fontsize=12)
    plt.tight_layout()
    save_fig(fig, out)
    print(f"  ✓ {out.name}  (position linear-probe R^2 = {r2:.3f})")
    return r2


# --------------------------------------------------------------------------- #
# Deliverable 2 — autoregressive rollout breakdown + decoded filmstrip
# --------------------------------------------------------------------------- #
@torch.no_grad()
def autoregressive_rollout(jepa, videos, device):
    """Roll the predictor on its own output from 2 GT frames.

    Returns (true_state, rolled_state) each [B, D, T, H, W]; rolled[:, :, t] for
    t >= 2 is produced autoregressively (the predictor never sees a true frame
    past index 1).
    """
    jepa.eval()
    state = jepa.encoder(videos.to(device))          # [B, D, T, H, W] (true future)
    T = state.shape[2]
    preds = state[:, :, :2].clone()                  # seed with 2 true latents
    for _ in range(2, T):
        nxt = jepa.predictor(preds[:, :, -2:], None)[:, :, -1:]  # predict next
        preds = torch.cat([preds, nxt], dim=2)
    return state, preds


@torch.no_grad()
def _latent_energy_per_frame(jepa, true_state, rolled_state):
    """Canonical latent prediction energy per frame: [T].

    Uses jepa.predcost (SquareLossSeq) — the SAME energy that defines the
    violation-of-expectation signal everywhere else — not a pixel/raw MSE.
    """
    T = true_state.shape[2]
    return np.array([
        jepa.predcost(true_state[:, :, t:t + 1], rolled_state[:, :, t:t + 1]).item()
        for t in range(T)
    ])


def fig_rollout_breakdown(jepa, decoder, device, out_dir, n_clips=16, T=16, seed=777):
    videos, _ = gen_plausible(n_clips, T, seed)
    true_state, rolled = autoregressive_rollout(jepa, videos, device)
    err = _latent_energy_per_frame(jepa, true_state, rolled)

    # Breakdown frame: first step whose error exceeds 2x the first prediction's.
    base = err[2] if len(err) > 2 else err[-1]
    knee = next((t for t in range(2, T) if err[t] > 2 * base), T - 1)

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=DPI)
    ax.plot(range(T), err, "o-", color="#1f77b4")
    ax.axvspan(0, 1.5, color="gray", alpha=0.15, label="2 GT seed frames")
    ax.axvline(knee, color="red", linestyle="--", label=f"breakdown ≈ frame {knee}")
    ax.axvline(10, color="green", linestyle=":", label="training horizon T=10")
    ax.set_xlabel("Frame index (autoregressive)")
    ax.set_ylabel("Latent prediction energy  (predcost vs true future)")
    ax.set_title("Where does open-loop (dream) rollout break down?")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = out_dir / "fig_rollout_breakdown.pdf"
    save_fig(fig, out)
    print(f"  ✓ {out.name}  (breakdown ≈ frame {knee})")

    if decoder is None:
        print("  ! no decoder in checkpoint — skipping filmstrip")
        return

    # Decoded filmstrip for one clip: true | recon(true latent) | predicted(rolled latent)
    with torch.no_grad():
        true_px = videos[0, 0].cpu().numpy()                          # [T, 64, 64]
        recon_px = decoder(true_state[:1])[0, 0].clamp(0, 1).cpu().numpy()
        pred_px = decoder(rolled[:1])[0, 0].clamp(0, 1).cpu().numpy()

    rows = [("true frames", true_px),
            ("decode(true latent)", recon_px),
            ("decode(rolled latent)", pred_px)]
    fig, axes = plt.subplots(3, T, figsize=(1.1 * T, 3.6), dpi=DPI)
    for r, (label, strip) in enumerate(rows):
        for t in range(T):
            ax = axes[r, t]
            ax.imshow(strip[t], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}", fontsize=7)
            if t == 0:
                ax.set_ylabel(label, fontsize=8)
        if r == 2:
            axes[r, knee].spines[:].set_color("red")
            for sp in axes[r, knee].spines.values():
                sp.set_linewidth(2)
    fig.suptitle("Open-loop prediction in pixels (rolled latents decoded)", fontsize=11)
    plt.tight_layout()
    out = out_dir / "fig_rollout_filmstrip.pdf"
    save_fig(fig, out)
    print(f"  ✓ {out.name}")


# --------------------------------------------------------------------------- #
# Deliverable 3 — why latent eval beats pixel-decode eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _per_frame_latent_energy(jepa, clip, device):
    """1-step projected latent energy per predicted frame for one clip [1,1,T,H,W]."""
    proj = jepa.predcost.proj
    state = jepa.encoder(clip.to(device))
    T = state.shape[2]
    out = []
    for t in range(T - 2):
        ctx = state[:, :, t:t + 2]
        pred = jepa.predictor(ctx, None)                  # [1, D, 1, H, W]
        tgt = state[:, :, t + 2:t + 3]
        a = pred.permute(0, 2, 3, 4, 1).reshape(-1, state.shape[1])
        b = tgt.permute(0, 2, 3, 4, 1).reshape(-1, state.shape[1])
        out.append((proj(a) - proj(b)).pow(2).mean().item())
    return np.array(out)                                  # predicts frames 2..T-1


@torch.no_grad()
def _per_frame_pixel_error(decoder, jepa, clip, device):
    """Per-frame pixel reconstruction MSE for one clip [1,1,T,H,W]: [T]."""
    clip = clip.to(device)
    recon = decoder(jepa.encoder(clip))
    return ((recon - clip) ** 2).mean(dim=[1, 3, 4])[0].cpu().numpy()


def fig_latent_vs_pixel(jepa, decoder, cfg, device, out_dir, viol="teleport"):
    nsteps = cfg.model.steps
    # --- per-frame: one matched pair ---
    pair1 = build_probe_pairs(n_pairs=1, T=cfg.data.T, seed=999, violations=(viol,))
    t_v = int(pair1[viol]["t_v"][0])
    lat_p = _per_frame_latent_energy(jepa, pair1[viol]["plausible"], device)
    lat_i = _per_frame_latent_energy(jepa, pair1[viol]["impossible"], device)
    pix_p = pix_i = None
    if decoder is not None:
        pix_p = _per_frame_pixel_error(decoder, jepa, pair1[viol]["plausible"], device)
        pix_i = _per_frame_pixel_error(decoder, jepa, pair1[viol]["impossible"], device)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=DPI)
    fr_lat = list(range(2, cfg.data.T))
    axes[0].plot(fr_lat, lat_p, "b-o", label="plausible")
    axes[0].plot(fr_lat, lat_i, "r-o", label="impossible")
    axes[0].axvline(t_v, color="gray", ls="--", label=f"t_v={t_v}")
    axes[0].set_title("LATENT prediction energy — spikes at violation")
    axes[0].set_xlabel("frame"); axes[0].set_ylabel("1-step latent energy")
    axes[0].legend(fontsize=8)

    if pix_p is not None:
        fr_px = list(range(cfg.data.T))
        axes[1].plot(fr_px, pix_p, "b-o", label="plausible")
        axes[1].plot(fr_px, pix_i, "r-o", label="impossible")
        axes[1].axvline(t_v, color="gray", ls="--", label=f"t_v={t_v}")
        axes[1].set_title("PIXEL reconstruction error — flat across violation")
        axes[1].set_xlabel("frame"); axes[1].set_ylabel("per-frame pixel MSE")
        axes[1].legend(fontsize=8)
    fig.suptitle("Why evaluate in latent space, not after decoding", fontsize=12)
    plt.tight_layout()
    out = out_dir / "fig_latent_vs_pixel_perframe.pdf"
    save_fig(fig, out)
    print(f"  ✓ {out.name}")

    if decoder is None:
        return
    # --- per-clip scatter: latent separates, pixel overlaps ---
    pairs = build_probe_pairs(n_pairs=150, T=cfg.data.T, seed=999, violations=(viol,))
    lat = lambda c: clip_energy(jepa, c, nsteps, device).numpy()
    pix = lambda c: clip_pixel_energy(decoder, jepa, c, device).numpy()
    lp, li = lat(pairs[viol]["plausible"]), lat(pairs[viol]["impossible"])
    pp, pi = pix(pairs[viol]["plausible"]), pix(pairs[viol]["impossible"])
    y = np.r_[np.zeros(len(lp)), np.ones(len(li))]
    auroc_lat = roc_auc_score(y, np.r_[lp, li])
    auroc_pix = roc_auc_score(y, np.r_[pp, pi])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    ax.scatter(pp, lp, c="#2166ac", s=14, label="plausible", alpha=0.7)
    ax.scatter(pi, li, c="#d6604d", s=14, label="impossible", alpha=0.7)
    ax.set_xlabel(f"pixel reconstruction MSE   (AUROC={auroc_pix:.2f})")
    ax.set_ylabel(f"latent prediction energy   (AUROC={auroc_lat:.2f})")
    ax.set_title(f"{viol}: latent separates the classes, pixel does not")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = out_dir / "fig_latent_vs_pixel_scatter.pdf"
    save_fig(fig, out)
    print(f"  ✓ {out.name}  (latent AUROC={auroc_lat:.2f} vs pixel AUROC={auroc_pix:.2f})")


# --------------------------------------------------------------------------- #
# Clustering analyses — does the latent space separate digit identity (10) and
# anomaly type (3)?  Embeddings via PCA / t-SNE / UMAP (3-D), clustering done in
# the RAW latent space (no dimensionality reduction).
# --------------------------------------------------------------------------- #
def load_mnist_labeled(split="test"):
    """MNIST glyphs [N,28,28] in [0,1] AND their integer labels [N], same order
    as stimuli.load_mnist_digits (torchvision dataset order)."""
    import os

    from torchvision.datasets import MNIST
    root = os.environ.get("EBJEPA_DSETS", os.path.join(os.getcwd(), "data"))
    ds = MNIST(root=root, train=(split == "train"), download=True)
    return ds.data.numpy().astype(np.float32) / 255.0, ds.targets.numpy()


def gen_plausible_labeled(n, T, seed, split="test"):
    """`n` plausible single-digit clips plus the digit class label per clip.

    Returns (videos [n,1,T,64,64], labels [n])."""
    digits, labels = load_mnist_labeled(split)
    rng = np.random.RandomState(seed)
    vids, ys = [], []
    for _ in range(n):
        idx = rng.randint(len(digits))
        pos = sample_plausible(rng, T)
        vids.append(render_clip(pos, digits[idx]))
        ys.append(int(labels[idx]))
    return torch.from_numpy(np.stack(vids)), np.array(ys)


@torch.no_grad()
def pooled_latents_perclip(jepa, videos, device, batch_size=32):
    """One latent vector per clip: mean over time AND space -> [n, D]."""
    jepa.eval()
    out = []
    for s in range(0, videos.shape[0], batch_size):
        state = jepa.encoder(videos[s:s + batch_size].to(device))  # [B,D,T,H,W]
        out.append(state.mean(dim=[2, 3, 4]).cpu())                # [B, D]
    return torch.cat(out).numpy()


@torch.no_grad()
def anomaly_energy_profiles(jepa, clips, device, batch_size=32):
    """Per-clip 1-step latent prediction-energy profile: [n, T-2].

    This is the 'latent representation of the anomaly' — the temporal surprise
    signature. Teleport, reversal and passthrough leave different shapes.
    """
    jepa.eval()
    proj = jepa.predcost.proj
    out = []
    for s in range(0, clips.shape[0], batch_size):
        state = jepa.encoder(clips[s:s + batch_size].to(device))  # [B,D,T,H,W]
        B, D, T, H, W = state.shape
        prof = torch.zeros(B, T - 2, device=device)
        for t in range(T - 2):
            pred = jepa.predictor(state[:, :, t:t + 2], None)      # [B,D,1,H,W]
            tgt = state[:, :, t + 2:t + 3]
            a = proj(pred.permute(0, 2, 3, 4, 1).reshape(-1, D))
            b = proj(tgt.permute(0, 2, 3, 4, 1).reshape(-1, D))
            prof[:, t] = (a - b).pow(2).mean(-1).reshape(B, H * W).mean(1)
        out.append(prof.cpu())
    return torch.cat(out).numpy()


def embed_3d(X):
    """Return {'PCA','t-SNE','UMAP'} each a [n,3] embedding of standardized X."""
    Xs = StandardScaler().fit_transform(X)
    perp = max(5, min(30, (len(Xs) - 1) // 3))
    n_nb = max(5, min(15, len(Xs) - 1))
    return {
        "PCA": PCA(n_components=3).fit_transform(Xs),
        "t-SNE": TSNE(n_components=3, perplexity=perp, init="pca",
                      learning_rate="auto", random_state=0).fit_transform(Xs),
        "UMAP": umap.UMAP(n_components=3, n_neighbors=n_nb, min_dist=0.1,
                          random_state=0).fit_transform(Xs),
    }


def cluster_report(X, y_true, k, name):
    """KMeans in the RAW latent space + supervised linear probe. Returns a dict."""
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    yp = km.labels_
    acc = cross_val_score(LogisticRegression(max_iter=2000), Xs, y_true,
                          cv=5, scoring="accuracy").mean()
    rep = {
        "ari": adjusted_rand_score(y_true, yp),
        "nmi": normalized_mutual_info_score(y_true, yp),
        "silhouette": silhouette_score(Xs, yp),
        "probe_acc": acc,
        "labels": yp,
    }
    print(f"  [{name}] k={k}  ARI={rep['ari']:.3f}  NMI={rep['nmi']:.3f}  "
          f"silhouette={rep['silhouette']:.3f}  linear-probe acc={acc:.3f} "
          f"(chance={1.0 / k:.3f})")
    return rep


def _scatter3d_panels(embeds, y, classes, title, out, palette_name):
    """One row of 3-D scatter panels (PCA | t-SNE | UMAP), colored by class."""
    palette = sns.color_palette(palette_name, len(classes))
    fig = plt.figure(figsize=(6 * len(embeds), 6), dpi=DPI)
    for i, (mname, E) in enumerate(embeds.items()):
        ax = fig.add_subplot(1, len(embeds), i + 1, projection="3d")
        for ci, c in enumerate(classes):
            m = y == c
            ax.scatter(E[m, 0], E[m, 1], E[m, 2], color=palette[ci],
                       label=str(c), s=14, alpha=0.7, depthshade=False)
        ax.set_title(mname)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="class",
               fontsize=9, markerscale=1.5)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    save_fig(fig, out)
    print(f"  ✓ {out.name}")


def _contingency_heatmap(y_true, y_pred, k, title, out, xlabel):
    """Seaborn heatmap of cluster (rows) vs true class (cols) counts."""
    classes = sorted(set(int(v) for v in y_true))
    M = np.zeros((k, len(classes)), dtype=int)
    for c, l in zip(y_pred, y_true):
        M[int(c), classes.index(int(l))] += 1
    fig, ax = plt.subplots(figsize=(0.8 * len(classes) + 3, 0.7 * k + 2), dpi=DPI)
    sns.heatmap(M, annot=True, fmt="d", cmap="mako", cbar=True,
                xticklabels=classes, yticklabels=range(k), ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("KMeans cluster")
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig, out)
    print(f"  ✓ {out.name}")


def fig_digit_clustering(jepa, device, out_dir, n_clips=500, T=10, seed=20):
    """Does the latent encode digit IDENTITY? 3-D PCA/t-SNE/UMAP + raw clustering."""
    videos, y = gen_plausible_labeled(n_clips, T, seed)
    X = pooled_latents_perclip(jepa, videos, device)          # [n, D]
    embeds = embed_3d(X)
    _scatter3d_panels(
        embeds, y, list(range(10)),
        "Per-clip latent colored by digit identity (0-9)",
        out_dir / "fig_digit_identity_3d.pdf", "tab10")
    rep = cluster_report(X, y, k=10, name="digit-identity")
    _contingency_heatmap(
        y, rep["labels"], 10,
        f"Digit clustering  (ARI={rep['ari']:.2f}, probe acc={rep['probe_acc']:.2f})",
        out_dir / "fig_digit_clustering.pdf", "true digit")


def fig_anomaly_clustering(jepa, cfg, device, out_dir, n_pairs=200):
    """Does the latent separate the 3 ANOMALY types? 3-D embeds + raw clustering."""
    pairs = build_probe_pairs(n_pairs=n_pairs, T=cfg.data.T, seed=999)
    clips, y = [], []
    for i, v in enumerate(VIOLATIONS):
        clips.append(pairs[v]["impossible"])
        y += [i] * pairs[v]["impossible"].shape[0]
    clips = torch.cat(clips)
    y = np.array(y)
    X = anomaly_energy_profiles(jepa, clips, device)          # [n, T-2]
    embeds = embed_3d(X)
    _scatter3d_panels(
        embeds, y, list(range(len(VIOLATIONS))),
        "Anomaly latent surprise-profile by violation type",
        out_dir / "fig_anomaly_3d.pdf", "Set1")
    rep = cluster_report(X, y, k=len(VIOLATIONS), name="anomaly-type")
    _contingency_heatmap(
        y, rep["labels"], len(VIOLATIONS),
        f"Anomaly clustering  (ARI={rep['ari']:.2f}, probe acc={rep['probe_acc']:.2f})",
        out_dir / "fig_anomaly_clustering.pdf", "true violation (0=tel,1=rev,2=pass)")


# --------------------------------------------------------------------------- #
# Normal vs anomaly — do anomalies live OFF the regular-digit manifold, or are
# they mixed in?  Same "info present but entangled?" question, applied to
# anomaly-vs-normal. Headline = kNN-distance / LOF AUROC, not KMeans/ARI.
# --------------------------------------------------------------------------- #
def _embed_2d_joint(Z_all):
    """PCA/t-SNE/UMAP (2-D) fit JOINTLY on standardized Z_all -> one frame each."""
    Xs = StandardScaler().fit_transform(Z_all)
    perp = max(5, min(30, (len(Xs) - 1) // 3))
    return {
        "PCA": PCA(n_components=2).fit_transform(Xs),
        "t-SNE": TSNE(n_components=2, perplexity=perp, init="pca",
                      learning_rate="auto", random_state=0).fit_transform(Xs),
        "UMAP": umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                          random_state=0).fit_transform(Xs),
    }


def _knn_dist_to_normal(Z_query, Z_normal, k=10, exclude_self=False):
    """Mean distance to the k nearest NORMAL points, in raw latent space."""
    nn = NearestNeighbors(n_neighbors=k + (1 if exclude_self else 0)).fit(Z_normal)
    d, _ = nn.kneighbors(Z_query)
    if exclude_self:
        d = d[:, 1:]                 # drop the self-match (distance 0)
    return d.mean(1)


def fig_normal_vs_anomaly(jepa, cfg, device, out_dir, n_normal=600, n_pairs=200):
    """One latent picture for normals + anomalies, with quantitative separation."""
    # --- assemble raw pooled latents in ONE representation ---
    v_norm, y_digit = gen_plausible_labeled(n_normal, cfg.data.T, seed=20)
    Z_normal = pooled_latents_perclip(jepa, v_norm, device)        # [Nn, D] raw
    pairs = build_probe_pairs(n_pairs=n_pairs, T=cfg.data.T, seed=999)
    anom_clips = torch.cat([pairs[v]["impossible"] for v in VIOLATIONS])
    Z_anom = pooled_latents_perclip(jepa, anom_clips, device)      # [Na, D] raw

    Z_all = np.concatenate([Z_normal, Z_anom])
    is_anom = np.r_[np.zeros(len(Z_normal)), np.ones(len(Z_anom))].astype(int)
    print(f"  Z_normal={Z_normal.shape}  Z_anom={Z_anom.shape}  D={Z_all.shape[1]} "
          "(raw pooled encoder latent, pre-projection; reducers fit jointly on Z_all)")

    embeds = _embed_2d_joint(Z_all)                                # joint frame

    # --- figure 1: colored by is_anom (normals grey, anomalies red on top) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=DPI)
    for ax, (mname, E) in zip(axes, embeds.items()):
        ax.scatter(E[is_anom == 0, 0], E[is_anom == 0, 1], c="lightgrey",
                   s=10, alpha=0.4, label="normal")
        ax.scatter(E[is_anom == 1, 0], E[is_anom == 1, 1], c="#d62728",
                   s=12, alpha=0.7, label="anomaly")
        ax.set_title(mname); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="best", fontsize=11)
    fig.suptitle("Normal digits vs anomalies in one latent frame", fontsize=15)
    plt.tight_layout()
    save_fig(fig, out_dir / "latent_normal_vs_anom.pdf")
    print("  ✓ latent_normal_vs_anom.png")

    # --- figure 2: UMAP, normals by digit class, anomalies as black x ---
    E = embeds["UMAP"]
    palette = sns.color_palette("tab10", 10)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI)
    for d in range(10):
        m = (is_anom == 0) & (np.r_[y_digit, -np.ones(len(Z_anom))] == d)
        ax.scatter(E[m, 0], E[m, 1], color=palette[d], s=12, alpha=0.6, label=str(d))
    ax.scatter(E[is_anom == 1, 0], E[is_anom == 1, 1], marker="x", c="black",
               s=22, alpha=0.8, label="anomaly")
    ax.set_title("UMAP — normals by digit, anomalies (x) overlaid")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    save_fig(fig, out_dir / "latent_combined_byclass.pdf")
    print("  ✓ latent_combined_byclass.png")

    # --- quantitative separation (raw latent space) ---
    d_norm = _knn_dist_to_normal(Z_normal, Z_normal, k=10, exclude_self=True)
    d_anom = _knn_dist_to_normal(Z_anom, Z_normal, k=10, exclude_self=False)
    y = np.r_[np.zeros(len(d_norm)), np.ones(len(d_anom))]
    auroc_knn = roc_auc_score(y, np.r_[d_norm, d_anom])

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Z_normal)
    s_norm = -lof.score_samples(Z_normal)
    s_anom = -lof.score_samples(Z_anom)
    auroc_lof = roc_auc_score(y, np.r_[s_norm, s_anom])

    def _mi(x):
        return np.median(x), np.percentile(x, 25), np.percentile(x, 75)

    mn, mn_lo, mn_hi = _mi(d_norm)
    ma, ma_lo, ma_hi = _mi(d_anom)

    # --- figure 3: anomaly-score distributions ---
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=DPI)
    lo, hi = min(d_norm.min(), d_anom.min()), max(d_norm.max(), d_anom.max())
    bins = np.linspace(lo, hi, 40)
    ax.hist(d_norm, bins=bins, alpha=0.6, color="grey", density=True, label="normal")
    ax.hist(d_anom, bins=bins, alpha=0.6, color="#d62728", density=True, label="anomaly")
    ax.set_xlabel("mean distance to 10 nearest NORMAL latents")
    ax.set_ylabel("density")
    ax.set_title(f"Distance-to-normal-manifold  (kNN AUROC={auroc_knn:.3f})")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, out_dir / "anomaly_score_hist.pdf")
    print("  ✓ anomaly_score_hist.png")

    print("  " + "=" * 64)
    print(f"  SEPARATION RESULTS (raw pooled encoder latent, D={Z_all.shape[1]})")
    print(f"  kNN-distance-to-normals AUROC = {auroc_knn:.3f}")
    print(f"  LOF (novelty) AUROC           = {auroc_lof:.3f}")
    print(f"  median dist-to-normal  normal = {mn:.4f}  IQR[{mn_lo:.4f}, {mn_hi:.4f}]")
    print(f"  median dist-to-normal  anomaly= {ma:.4f}  IQR[{ma_lo:.4f}, {ma_hi:.4f}]")
    verdict = "ARE" if auroc_knn >= 0.7 else "ARE NOT"
    print(f"  >>> Anomalies {verdict} well separated from regular digits, "
          f"AUROC={auroc_knn:.3f}")
    print("  " + "=" * 64)


# --------------------------------------------------------------------------- #
# 3-D UMAP n_neighbors sweep + density clustering (HDBSCAN). The sweep is scored
# (5-NN accuracy in-embedding) so "best n" is a number, not a vibe. Clustering
# is done in the RAW latent AND on the best 3-D UMAP — never KMeans here.
# --------------------------------------------------------------------------- #
UMAP_NEIGHBORS = [2, 3, 5, 8, 10, 12, 15, 20, 25, 30,
                  40, 50, 60, 75, 100, 125, 150, 200, 300, 400]


def _knn_acc(E, y, k=5):
    """5-fold CV 5-NN classification accuracy in an embedding E."""
    k = min(k, np.bincount(y).min() - 1) if np.bincount(y).min() > 1 else 1
    k = max(1, k)
    return cross_val_score(KNeighborsClassifier(n_neighbors=k), E, y,
                           cv=5, scoring="accuracy").mean()


def fig_umap_sweep_3d(Z, y, classes, out, title, palette_name):
    """4x5 grid of 3-D UMAPs of raw Z over n_neighbors, each scored by 5-NN acc.

    Returns (best_n, best_embedding) so clustering can reuse the best 3-D UMAP.
    """
    Xs = StandardScaler().fit_transform(Z)
    palette = sns.color_palette(palette_name, len(classes))
    n_max = len(Xs) - 1
    fig = plt.figure(figsize=(25, 20), dpi=120)

    scores, embeds = [], []
    for K in UMAP_NEIGHBORS:
        k = min(K, n_max)
        E = umap.UMAP(n_components=3, n_neighbors=k, min_dist=0.1,
                      metric="euclidean", random_state=0).fit_transform(Xs)
        scores.append(_knn_acc(E, y))
        embeds.append(E)
    best_i = int(np.argmax(scores))

    for i, (K, E, sc) in enumerate(zip(UMAP_NEIGHBORS, embeds, scores)):
        ax = fig.add_subplot(4, 5, i + 1, projection="3d")
        for ci, c in enumerate(classes):
            m = y == c
            ax.scatter(E[m, 0], E[m, 1], E[m, 2], color=palette[ci], s=6,
                       alpha=0.6, depthshade=False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
        t = ax.set_title(f"n={K} | kNN={sc:.2f}", fontsize=12)
        if i == best_i:
            t.set_color("red"); t.set_fontweight("bold")
            for sp in ax.spines.values():
                sp.set_edgecolor("red"); sp.set_linewidth(2)
    fig.suptitle(f"{title}  —  best n={UMAP_NEIGHBORS[best_i]} "
                 f"(kNN={scores[best_i]:.2f}, raw {Z.shape[1]}-D latent)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, out)
    print(f"  ✓ {out.name}  (best n_neighbors={UMAP_NEIGHBORS[best_i]}, "
          f"kNN={scores[best_i]:.3f})")
    return UMAP_NEIGHBORS[best_i], embeds[best_i]


def _clustering_metrics(X, y_true, labels):
    """(#clusters, noise_frac, ari, nmi, silhouette) for a labelling with -1=noise."""
    uniq = set(labels)
    n_clusters = len(uniq) - (1 if -1 in uniq else 0)
    noise = float(np.mean(labels == -1))
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    sil = float("nan")
    mask = labels != -1
    if mask.sum() > 2 and len(set(labels[mask])) >= 2:
        sil = silhouette_score(X[mask], labels[mask])
    return n_clusters, noise, ari, nmi, sil


def hdbscan_report(Z_raw, E_umap, y_true, name, min_cluster_size=20):
    """HDBSCAN on (a) the full-D raw latent and (b) the best 3-D UMAP.

    Prints metrics and returns {'raw': labels, 'umap': labels} so the cluster
    assignments can be visualized.
    """
    k_classes = len(set(int(v) for v in y_true))
    knn_head = _knn_acc(StandardScaler().fit_transform(Z_raw), y_true)
    print(f"  [{name}] kNN headline (raw {Z_raw.shape[1]}-D) = {knn_head:.3f} "
          f"(chance={1.0 / k_classes:.3f}, true #classes={k_classes})")

    out = {}
    for key, space, X in [
        ("raw", f"raw {Z_raw.shape[1]}-D latent", StandardScaler().fit_transform(Z_raw)),
        ("umap", "best 3-D UMAP", E_umap),
    ]:
        lab = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
        nc, noise, ari, nmi, sil = _clustering_metrics(X, y_true, lab)
        print(f"    HDBSCAN on {space:24s}: #clusters={nc}  noise={noise:.2f}  "
              f"ARI={ari:.3f}  NMI={nmi:.3f}  silhouette={sil:.3f}")
        out[key] = lab
    return out


def _scatter3d_labels(ax, E, labels, palette_name=None):
    """3-D scatter colored by integer cluster labels; -1 (noise) drawn grey."""
    uniq = sorted(set(int(v) for v in labels))
    non_noise = [u for u in uniq if u != -1]
    pal = sns.color_palette(palette_name or "husl", max(1, len(non_noise)))
    ci = 0
    for u in uniq:
        m = labels == u
        if u == -1:
            ax.scatter(E[m, 0], E[m, 1], E[m, 2], color="lightgrey", s=6,
                       alpha=0.35, depthshade=False, label="noise")
        else:
            ax.scatter(E[m, 0], E[m, 1], E[m, 2], color=pal[ci], s=6,
                       alpha=0.75, depthshade=False)
            ci += 1
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)


def fig_hdbscan_view(E_umap, y_true, labs, classes, out, title, palette_name):
    """Best 3-D UMAP shown 3 ways: true labels | HDBSCAN(raw) | HDBSCAN(UMAP)."""
    truth_pal = sns.color_palette(palette_name, len(classes))
    fig = plt.figure(figsize=(21, 7), dpi=DPI)

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for ci, c in enumerate(classes):
        m = y_true == c
        ax.scatter(E_umap[m, 0], E_umap[m, 1], E_umap[m, 2], color=truth_pal[ci],
                   s=6, alpha=0.7, depthshade=False, label=str(c))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
    ax.set_title("true labels")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    _scatter3d_labels(ax, E_umap, labs["raw"])
    nc = len(set(labs["raw"])) - (1 if -1 in labs["raw"] else 0)
    ax.set_title(f"HDBSCAN on raw latent ({nc} clusters)")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    _scatter3d_labels(ax, E_umap, labs["umap"])
    nc = len(set(labs["umap"])) - (1 if -1 in labs["umap"] else 0)
    ax.set_title(f"HDBSCAN on UMAP ({nc} clusters)")

    fig.suptitle(f"{title}  —  cluster assignments vs truth (grey = noise)",
                 fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out)
    print(f"  ✓ {out.name}")


def fig_umap_sweep_and_hdbscan(jepa, cfg, device, out_dir):
    """Run the 3-D UMAP sweep + HDBSCAN for BOTH the digit and anomaly latents."""
    # digit latent: pooled raw encoder latent, 10 classes
    v, yd = gen_plausible_labeled(500, cfg.data.T, seed=20)
    Zd = pooled_latents_perclip(jepa, v, device)
    # anomaly latent: surprise-profile, 3 classes
    pairs = build_probe_pairs(n_pairs=200, T=cfg.data.T, seed=999)
    aclips = torch.cat([pairs[viol]["impossible"] for viol in VIOLATIONS])
    ya = np.concatenate([[i] * pairs[viol]["impossible"].shape[0]
                         for i, viol in enumerate(VIOLATIONS)])
    Za = anomaly_energy_profiles(jepa, aclips, device)

    print("  -- digit latent (10 classes) --")
    bn_d, E_d = fig_umap_sweep_3d(Zd, yd, list(range(10)),
                                  out_dir / "umap_sweep_digit.pdf",
                                  "Digit identity 3-D UMAP sweep", "tab10")
    labs_d = hdbscan_report(Zd, E_d, yd, "digit", min_cluster_size=15)
    fig_hdbscan_view(E_d, yd, labs_d, list(range(10)),
                     out_dir / "hdbscan_digit_3d.pdf",
                     "Digit latent — HDBSCAN", "tab10")

    print("  -- anomaly latent (3 classes) --")
    bn_a, E_a = fig_umap_sweep_3d(Za, ya, list(range(len(VIOLATIONS))),
                                  out_dir / "umap_sweep_anomaly.pdf",
                                  "Anomaly type 3-D UMAP sweep", "Set1")
    labs_a = hdbscan_report(Za, E_a, ya, "anomaly", min_cluster_size=30)
    fig_hdbscan_view(E_a, ya, labs_a, list(range(len(VIOLATIONS))),
                     out_dir / "hdbscan_anomaly_3d.pdf",
                     "Anomaly latent — HDBSCAN", "Set1")


# --------------------------------------------------------------------------- #
# Anomaly-only parameter sweep: 20 2-D embeddings (PCA / t-SNE / UMAP variants)
# of the anomaly latent, colored by violation type, to eyeball whether ANY
# configuration isolates the 3 types as defined clusters. Each panel scored by
# 5-NN accuracy in-embedding; best highlighted.
# --------------------------------------------------------------------------- #
def fig_anomaly_param_sweep(jepa, cfg, device, out_dir, n_pairs=200):
    pairs = build_probe_pairs(n_pairs=n_pairs, T=cfg.data.T, seed=999)
    aclips = torch.cat([pairs[v]["impossible"] for v in VIOLATIONS])
    y = np.concatenate([[i] * pairs[v]["impossible"].shape[0]
                        for i, v in enumerate(VIOLATIONS)])
    Z = anomaly_energy_profiles(jepa, aclips, device)             # [N, T-2]
    Xs = StandardScaler().fit_transform(Z)
    print(f"  anomaly latent Z={Z.shape} (surprise-profile), 3 violation types")

    configs = []
    P = PCA(n_components=min(4, Xs.shape[1])).fit_transform(Xs)
    for (i, j) in [(0, 1), (0, 2), (1, 2), (2, 3)]:
        configs.append((f"PCA {i + 1}-{j + 1}", P[:, [i, j]]))
    for perp in [5, 10, 15, 20, 30, 40, 50, 75]:
        E = TSNE(n_components=2, perplexity=perp, init="pca",
                 learning_rate="auto", random_state=0).fit_transform(Xs)
        configs.append((f"t-SNE perp={perp}", E))
    for nn_, md in [(5, 0.0), (5, 0.1), (15, 0.0), (15, 0.1),
                    (30, 0.1), (30, 0.5), (50, 0.1), (50, 0.5)]:
        E = umap.UMAP(n_components=2, n_neighbors=nn_, min_dist=md,
                      metric="euclidean", random_state=0).fit_transform(Xs)
        configs.append((f"UMAP n={nn_},md={md}", E))

    scores = [_knn_acc(E, y) for _, E in configs]
    best_i = int(np.argmax(scores))
    palette = sns.color_palette("Set1", len(VIOLATIONS))

    fig, axes = plt.subplots(4, 5, figsize=(25, 20), dpi=120)
    for ax, (lab, E), sc, idx in zip(axes.ravel(), configs, scores,
                                     range(len(configs))):
        for ci in range(len(VIOLATIONS)):
            m = y == ci
            ax.scatter(E[m, 0], E[m, 1], color=palette[ci], s=8, alpha=0.6,
                       label=VIOLATIONS[ci])
        ax.set_xticks([]); ax.set_yticks([])
        t = ax.set_title(f"{lab} | kNN={sc:.2f}", fontsize=12)
        if idx == best_i:
            t.set_color("red"); t.set_fontweight("bold")
            for sp in ax.spines.values():
                sp.set_edgecolor("red"); sp.set_linewidth(2.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=13,
               title="violation", markerscale=2)
    fig.suptitle("Anomaly-only embeddings — does any config isolate the 3 "
                 f"violation types?  best: {configs[best_i][0]} "
                 f"(kNN={scores[best_i]:.2f})", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, out_dir / "anomaly_param_sweep.pdf")
    print(f"  ✓ anomaly_param_sweep.png  (best={configs[best_i][0]}, "
          f"kNN={scores[best_i]:.3f})")


# --------------------------------------------------------------------------- #
# Digit classification IN LATENT SPACE — train supervised heads on clean (no
# anomaly) clips and test on held-out clips. Answers "can the embedding classify
# 0-9?" directly, with test accuracy + a confusion matrix (not ARI).
# --------------------------------------------------------------------------- #
def fig_digit_classifier(jepa, cfg, device, out_dir, n_clips=2000, seed=20):
    videos, y = gen_plausible_labeled(n_clips, cfg.data.T, seed)   # clean digits
    emb = pooled_latents(jepa, videos, device)                     # [N, T, D]
    feats = {
        "pooled (D)": emb.mean(1),                                 # [N, D]
        "per-frame (T*D)": emb.reshape(len(emb), -1),              # [N, T*D]
    }
    classifiers = {
        "linear (LogReg)": lambda: LogisticRegression(max_iter=3000),
        "kNN-5": lambda: KNeighborsClassifier(n_neighbors=5),
        "MLP (128,64)": lambda: MLPClassifier(hidden_layer_sizes=(128, 64),
                                              max_iter=800, early_stopping=True),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300,
                                                       random_state=0),
    }

    results = {}      # feat_name -> {clf_name -> acc}
    best = (None, None, -1.0, None, None)  # feat, clf, acc, y_test, y_pred
    print("  classifier test accuracy (chance = 0.100, held-out clips):")
    for fname, X in feats.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              stratify=y, random_state=0)
        results[fname] = {}
        for cname, make in classifiers.items():
            clf = make_pipeline(StandardScaler(), make())
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
            acc = accuracy_score(yte, yp)
            results[fname][cname] = acc
            print(f"    {fname:16s} | {cname:16s} acc = {acc:.3f}")
            if acc > best[2]:
                best = (fname, cname, acc, yte, yp)

    # --- figure: accuracy bars + confusion matrix of the best model ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=DPI)
    fnames = list(results)
    cnames = list(classifiers)
    width = 0.8 / len(fnames)
    x = np.arange(len(cnames))
    for i, fn in enumerate(fnames):
        ax1.bar(x + i * width, [results[fn][c] for c in cnames], width, label=fn)
    ax1.axhline(0.1, color="gray", ls="--", label="chance (0.10)")
    ax1.set_xticks(x + width * (len(fnames) - 1) / 2)
    ax1.set_xticklabels(cnames, rotation=20, ha="right", fontsize=10)
    ax1.set_ylabel("held-out accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title("Digit classification from the latent (clean clips)")
    ax1.legend(fontsize=9)

    cm = confusion_matrix(best[3], best[4], normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis", cbar=True,
                xticklabels=range(10), yticklabels=range(10), ax=ax2,
                annot_kws={"size": 7})
    ax2.set_xlabel("predicted digit"); ax2.set_ylabel("true digit")
    ax2.set_title(f"Best: {best[1]} on {best[0]}  (acc={best[2]:.2f})")
    plt.tight_layout()
    save_fig(fig, out_dir / "fig_digit_classifier.pdf")
    print(f"  ✓ fig_digit_classifier.png  "
          f"(best {best[1]} on {best[0]} = {best[2]:.3f})")


# --------------------------------------------------------------------------- #
# Anomaly-TYPE classification in latent space — can a supervised head read off
# teleport/reversal/passthrough from the embedding's shape? Tested across three
# representations: surprise-profile, pooled encoder latent, full-latent PCA-50.
# --------------------------------------------------------------------------- #
def _classify_suite(feats, y, class_names, chance, title, out, palette_name):
    """Train LogReg/kNN/MLP/RF on each feature set; held-out acc + confusion fig."""
    classifiers = {
        "linear (LogReg)": lambda: LogisticRegression(max_iter=3000),
        "kNN-5": lambda: KNeighborsClassifier(n_neighbors=5),
        "MLP (128,64)": lambda: MLPClassifier(hidden_layer_sizes=(128, 64),
                                              max_iter=800, early_stopping=True),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300,
                                                       random_state=0),
    }
    results, best = {}, (None, None, -1.0, None, None)
    print(f"  classifier test accuracy (chance = {chance:.3f}, held-out clips):")
    for fname, X in feats.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              stratify=y, random_state=0)
        results[fname] = {}
        for cname, make in classifiers.items():
            clf = make_pipeline(StandardScaler(), make()).fit(Xtr, ytr)
            acc = accuracy_score(yte, clf.predict(Xte))
            results[fname][cname] = acc
            print(f"    {fname:22s} | {cname:16s} acc = {acc:.3f}")
            if acc > best[2]:
                best = (fname, cname, acc, yte, clf.predict(Xte))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=DPI)
    fnames, cnames = list(results), list(classifiers)
    width = 0.8 / len(fnames)
    x = np.arange(len(cnames))
    for i, fn in enumerate(fnames):
        ax1.bar(x + i * width, [results[fn][c] for c in cnames], width, label=fn)
    ax1.axhline(chance, color="gray", ls="--", label=f"chance ({chance:.2f})")
    ax1.set_xticks(x + width * (len(fnames) - 1) / 2)
    ax1.set_xticklabels(cnames, rotation=20, ha="right", fontsize=10)
    ax1.set_ylabel("held-out accuracy"); ax1.set_ylim(0, 1)
    ax1.set_title(title); ax1.legend(fontsize=9)

    cm = confusion_matrix(best[3], best[4], normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel("predicted"); ax2.set_ylabel("true")
    ax2.set_title(f"Best: {best[1]} on {best[0]}  (acc={best[2]:.2f})")
    plt.tight_layout()
    save_fig(fig, out)
    print(f"  ✓ {out.name}  (best {best[1]} on {best[0]} = {best[2]:.3f})")


def fig_anomaly_classifier(jepa, cfg, device, out_dir, n_pairs=300):
    pairs = build_probe_pairs(n_pairs=n_pairs, T=cfg.data.T, seed=999)
    clips = torch.cat([pairs[v]["impossible"] for v in VIOLATIONS])
    y = np.concatenate([[i] * pairs[v]["impossible"].shape[0]
                        for i, v in enumerate(VIOLATIONS)])
    surprise = anomaly_energy_profiles(jepa, clips, device)          # [N, T-2]
    pooled = pooled_latents_perclip(jepa, clips, device)             # [N, D]
    full = full_latent_flat(jepa, clips, device)                     # [N, 16*T*H*W]
    full50 = PCA(n_components=50, svd_solver="randomized",
                 random_state=0).fit_transform(full)
    feats = {
        "surprise-profile": surprise,
        "pooled latent (D)": pooled,
        "full-latent PCA-50": full50,
    }
    _classify_suite(feats, y, list(VIOLATIONS), 1.0 / len(VIOLATIONS),
                    "Anomaly-type classification from the latent",
                    out_dir / "fig_anomaly_classifier.pdf", "Set1")


# --------------------------------------------------------------------------- #
# HDBSCAN on the WHOLE latent (full [16,T,64,64] flattened -> PCA-50 -> HDBSCAN),
# with and without anomalies. Tests whether dense clusters exist in the full
# (un-pooled) space and whether anomalies separate there.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def full_latent_flat(jepa, videos, device, batch_size=16):
    """Full encoder latent per clip, flattened: [N, 16*T*H*W] (no pooling)."""
    jepa.eval()
    out = []
    for s in range(0, videos.shape[0], batch_size):
        st = jepa.encoder(videos[s:s + batch_size].to(device))   # [B,16,T,64,64]
        out.append(st.reshape(st.shape[0], -1).cpu().numpy().astype("float32"))
    return np.concatenate(out)


def _pca50_hdbscan_umap(Zfull, min_cluster_size, n_pca=50):
    """PCA-50 of full latent -> standardize -> HDBSCAN(raw50) + UMAP-3D + HDBSCAN(umap)."""
    P = PCA(n_components=min(n_pca, Zfull.shape[0] - 1, Zfull.shape[1]),
            svd_solver="randomized", random_state=0).fit_transform(Zfull)
    Ps = StandardScaler().fit_transform(P)
    lab_raw = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(Ps)
    E = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                  random_state=0).fit_transform(Ps)
    lab_umap = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(E)
    return Ps, E, {"raw": lab_raw, "umap": lab_umap}


def _report_full(name, Ps, E, y_true, labs):
    knn = _knn_acc(Ps, y_true)
    kc = len(set(int(v) for v in y_true))
    print(f"  [{name}] kNN headline (PCA-50 of full latent) = {knn:.3f} "
          f"(chance={1.0 / kc:.3f})")
    for key, space, X in [("raw", "full-latent PCA-50", Ps), ("umap", "UMAP-3D", E)]:
        nc, noise, ari, nmi, sil = _clustering_metrics(X, y_true, labs[key])
        print(f"    HDBSCAN on {space:20s}: #clusters={nc}  noise={noise:.2f}  "
              f"ARI={ari:.3f}  NMI={nmi:.3f}  silhouette={sil:.3f}")


def fig_full_latent_hdbscan(jepa, cfg, device, out_dir, n_normal=500, n_pairs=150):
    # ----- WITHOUT anomalies: full latent of clean digit clips -----
    v, yd = gen_plausible_labeled(n_normal, cfg.data.T, seed=20)
    Zn = full_latent_flat(jepa, v, device)
    print(f"  full latent (no anomalies): {Zn.shape}")
    Ps, E, labs = _pca50_hdbscan_umap(Zn, min_cluster_size=15)
    _report_full("digit/full", Ps, E, yd, labs)
    fig_hdbscan_view(E, yd, labs, list(range(10)),
                     out_dir / "full_latent_hdbscan_clean.pdf",
                     "WHOLE latent (no anomalies) — HDBSCAN vs digit", "tab10")

    # ----- WITH anomalies: clean digits + all 3 violation types -----
    pairs = build_probe_pairs(n_pairs=n_pairs, T=cfg.data.T, seed=999)
    aclips = torch.cat([pairs[viol]["impossible"] for viol in VIOLATIONS])
    Za = full_latent_flat(jepa, aclips, device)
    Zall = np.concatenate([Zn, Za])
    is_anom = np.r_[np.zeros(len(Zn)), np.ones(len(Za))].astype(int)
    print(f"  full latent (with anomalies): {Zall.shape}  "
          f"({len(Zn)} normal + {len(Za)} anomaly)")
    Ps2, E2, labs2 = _pca50_hdbscan_umap(Zall, min_cluster_size=20)
    _report_full("anom/full", Ps2, E2, is_anom, labs2)

    # anomaly purity of HDBSCAN clusters
    for key in ("raw", "umap"):
        lab = labs2[key]
        purities = [max(is_anom[lab == c].mean(), 1 - is_anom[lab == c].mean())
                    for c in sorted(set(lab)) if c != -1]
        if purities:
            print(f"    cluster purity ({key}): max={max(purities):.2f}  "
                  f"#clusters>=0.9-pure={sum(p >= 0.9 for p in purities)}/{len(purities)}")
    fig_hdbscan_view(E2, is_anom, labs2, [0, 1],
                     out_dir / "full_latent_hdbscan_withanom.pdf",
                     "WHOLE latent (with anomalies) — HDBSCAN vs is_anom", "Set1")


# --------------------------------------------------------------------------- #
def main():
    args = sys.argv[1:]
    if "--ckpt" not in args:
        raise SystemExit(
            "usage: python -m examples.intuitive_physics.visualize_latent "
            "--ckpt <path> [--out <dir>] [--fname cfgs/eval.yaml]")

    def _get(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    ckpt = _get("--ckpt")
    fname = _get("--fname", "examples/intuitive_physics/cfgs/eval.yaml")
    out_dir = Path(_get("--out", str(Path(ckpt).parent / "viz")))
    analysis = _get("--analysis", "all")  # all | basic | cluster
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(fname)
    device = setup_device(cfg.meta.get("device", "auto"))

    jepa = build_jepa(cfg, device)
    load_checkpoint(ckpt, jepa, device=device)
    jepa.eval()

    raw = torch.load(ckpt, map_location=device, weights_only=False)
    decoder = None
    if "decoder_state_dict" in raw:
        decoder = ImageDecoder(cfg.model.dstc, cfg.model.dobs, hidden_dim=cfg.model.dstc)
        decoder.load_state_dict(raw["decoder_state_dict"])
        decoder = decoder.to(device).eval()

    print(f"Writing visualizations to {out_dir}/  (analysis={analysis})")
    if analysis in ("all", "basic"):
        print("[basic 1/3] latent position map")
        fig_latent_position_map(jepa, device, out_dir / "fig_latent_position_map.pdf", T=cfg.data.T)
        print("[basic 2/3] autoregressive rollout breakdown")
        fig_rollout_breakdown(jepa, decoder, device, out_dir)
        print("[basic 3/3] latent vs pixel evaluation")
        fig_latent_vs_pixel(jepa, decoder, cfg, device, out_dir)
    if analysis in ("all", "cluster"):
        print("[cluster 1/2] digit-identity clustering (10 classes)")
        fig_digit_clustering(jepa, device, out_dir, T=cfg.data.T)
        print("[cluster 2/2] anomaly-type clustering (3 classes)")
        fig_anomaly_clustering(jepa, cfg, device, out_dir)
    if analysis in ("all", "separation"):
        print("[separation] normal digits vs anomalies in one latent frame")
        fig_normal_vs_anomaly(jepa, cfg, device, out_dir)
    if analysis in ("all", "sweep"):
        print("[sweep] 3-D UMAP n_neighbors sweep + HDBSCAN (digit & anomaly)")
        fig_umap_sweep_and_hdbscan(jepa, cfg, device, out_dir)
    if analysis in ("all", "anomsweep"):
        print("[anomsweep] 20 anomaly-only PCA/t-SNE/UMAP param variations")
        fig_anomaly_param_sweep(jepa, cfg, device, out_dir)
    if analysis in ("all", "classify"):
        print("[classify] supervised digit classification in latent space")
        fig_digit_classifier(jepa, cfg, device, out_dir)
    if analysis in ("all", "classify-anom"):
        print("[classify-anom] supervised anomaly-type classification in latent space")
        fig_anomaly_classifier(jepa, cfg, device, out_dir)
    if analysis in ("all", "fulldbscan"):
        print("[fulldbscan] HDBSCAN on the WHOLE latent (PCA-50), with/without anomalies")
        fig_full_latent_hdbscan(jepa, cfg, device, out_dir)
    print(f"\nDone -> {out_dir}/")


if __name__ == "__main__":
    main()
