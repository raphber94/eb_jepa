"""EEG — visualizations to understand the data and what the encoder learns.

Figures, each tied to a result of the TUAB track:

  signals    raw 19-channel windows, normal vs abnormal     (what the data is)
  psd        median PSD per class (25-75 pct band)           (WHY the random floor
  bandpower  mean band-power per class (delta..gamma)         is high: band-power
                                                              is linearly decodable)
  latent     2D PCA + t-SNE of frozen embeddings, TRAINED    (unsupervised view:
             vs RANDOM, colored by label                      often NOT separated)
  lda        1D LDA projection per class, TRAINED vs RANDOM  (the discriminant axis
                                                              the probe actually uses)
  scores     histogram of probe P(abnormal) per class        (makes AUROC tangible)
  auroc      bar chart AUROC: Random vs VICReg x win/rec      (the before/after)

Run (same runtime as eval.py — reads EDF + a checkpoint):
  python -m examples.eeg.viz --ckpt <.../latest.pth.tar> --what all --out eeg_figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.signal import welch

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder
from examples.eeg.eval import extract_features, extract_window_features, probe

BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 13),
         ("beta", 13, 30), ("gamma", 30, 50)]
CLS = [(0, "normal", "tab:blue"), (1, "abnormal", "tab:red")]


# --------------------------------------------------------------------------- #
# raw windows per class (signals / psd / bandpower)
# --------------------------------------------------------------------------- #
def _collect_windows(split, max_per_class):
    ds = EEGDataset(EEGConfig(split=split, mode="probe"))
    out = {0: [], 1: []}
    for i in range(len(ds)):
        wins, label, ok = ds[i]
        if not ok or len(out[label]) >= max_per_class:
            if all(len(v) >= max_per_class for v in out.values()):
                break
            continue
        out[label].append(wins.numpy())
    return {k: (np.concatenate(v, 0) if v else np.empty((0,))) for k, v in out.items()}


def fig_signals(out, sfreq):
    wins = _collect_windows("eval", 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, (label, name, _) in zip(axes, CLS):
        x = wins[label][0]
        offsets = np.arange(x.shape[0])[:, None] * 6.0
        ax.plot((x + offsets).T, lw=0.5, color="k")
        ax.set_title(f"{name.upper()} — one 10s window (19 ch)")
        ax.set_xlabel("samples"); ax.set_yticks([])
    fig.suptitle("Raw z-scored EEG windows")
    fig.tight_layout()
    _save(fig, out, "signals.png")


def _psd_per_class(split, sfreq, n=300):
    wins = _collect_windows(split, n)
    res = {}
    for label, name, _ in CLS:
        x = wins[label]
        f, pxx = welch(x, fs=sfreq, nperseg=min(512, x.shape[-1]), axis=-1)
        res[label] = (f, pxx.mean(axis=1))   # mean over channels -> [n, F]
    return res


def fig_psd(out, sfreq):
    res = _psd_per_class("eval", sfreq)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, name, col in CLS:
        f, pxx = res[label]
        m = f <= 50
        med = np.median(pxx, 0); lo = np.percentile(pxx, 25, 0); hi = np.percentile(pxx, 75, 0)
        ax.semilogy(f[m], med[m], color=col, label=f"{name} (n={pxx.shape[0]})")
        ax.fill_between(f[m], lo[m], hi[m], color=col, alpha=0.2)   # 25-75 pct, always >0
    for _, a, b in BANDS:
        ax.axvspan(a, b, color="k", alpha=0.03)
        ax.text((a + b) / 2, ax.get_ylim()[1], _, ha="center", va="top", fontsize=8, color="0.4")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD (log, median +/- IQR)")
    ax.set_title("Median PSD per class — normal keeps more alpha/beta power")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "psd.png")


def fig_bandpower(out, sfreq):
    res = _psd_per_class("eval", sfreq)
    names = [b[0] for b in BANDS]
    x = np.arange(len(BANDS)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, (label, cname, col) in enumerate(CLS):
        f, pxx = res[label]
        vals = []
        for _, a, b in BANDS:
            m = (f >= a) & (f < b)
            bp = pxx[:, m].sum(1)                # band-power per window
            vals.append(bp)
        med = [np.median(v) for v in vals]
        err = [[np.median(v) - np.percentile(v, 25) for v in vals],
               [np.percentile(v, 75) - np.median(v) for v in vals]]
        ax.bar(x + (k - 0.5) * w, med, w, yerr=err, capsize=3, color=col, label=cname, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("band-power (median +/- IQR)")
    ax.set_title("Per-band power by class — the linearly-decodable signal")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "bandpower.png")


# --------------------------------------------------------------------------- #
# latent / lda / scores / auroc  (share a feature cache over both encoders)
# --------------------------------------------------------------------------- #
def _feature_cache(cfg, state, device):
    cache = {}
    for name, load in [("VICReg", True), ("Random", False)]:
        enc = build_encoder(cfg.model).to(device)
        if load:
            enc.load_state_dict(state["encoder"])
        enc.eval()
        cache[name] = {
            "rec": (extract_features(enc, "train", device), extract_features(enc, "eval", device)),
            "win": (extract_window_features(enc, "train", device),
                    extract_window_features(enc, "eval", device)),
        }
    return cache


def fig_latent(out, cache):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for r, name in enumerate(["VICReg", "Random"]):
        (Xtr, ytr), (Xev, yev) = cache[name]["rec"]
        Z = StandardScaler().fit(Xtr).transform(Xev)
        for c, (pn, proj) in enumerate([
                ("PCA", PCA(2)),
                ("t-SNE", TSNE(2, init="pca", perplexity=30, random_state=0))]):
            P = proj.fit_transform(Z)
            ax = axes[r, c]
            ax.scatter(P[:, 0], P[:, 1], c=np.where(yev == 1, "tab:red", "tab:blue"), s=12, alpha=0.7)
            ax.set_title(f"{name} — {pn} (held-out)"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Frozen embeddings (eval) — blue=normal, red=abnormal")
    fig.tight_layout()
    _save(fig, out, "latent.png")


def fig_lda(out, cache):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, name in zip(axes, ["VICReg", "Random"]):
        (Xtr, ytr), (Xev, yev) = cache[name]["rec"]
        sc = StandardScaler().fit(Xtr)
        lda = LDA(n_components=1).fit(sc.transform(Xtr), ytr)
        p = lda.transform(sc.transform(Xev)).ravel()
        for label, cname, col in CLS:
            ax.hist(p[yev == label], bins=25, alpha=0.6, color=col, label=cname)
        ax.set_title(f"{name} — LDA axis (held-out)"); ax.set_xlabel("LDA projection"); ax.legend()
    fig.suptitle("Discriminant axis the probe uses — separation the 2D plots hide")
    fig.tight_layout()
    _save(fig, out, "lda.png")


def fig_scores(out, cache):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    (Xtr, ytr), (Xev, yev) = cache["VICReg"]["rec"]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xev))[:, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, cname, col in CLS:
        ax.hist(proba[yev == label], bins=25, alpha=0.6, color=col, label=cname)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("probe P(abnormal)"); ax.set_ylabel("# recordings (held-out)")
    ax.set_title("Linear-probe scores by class (VICReg, per-recording)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "scores.png")


def fig_auroc(out, cache):
    groups = [("per-window", "win"), ("per-recording", "rec")]
    encoders = ["Random", "VICReg"]
    vals = {e: [] for e in encoders}
    for e in encoders:
        for _, key in groups:
            (Xtr, ytr), (Xev, yev) = cache[e][key]
            vals[e].append(probe(Xtr, ytr, Xev, yev)["auroc"])
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, (e, col) in enumerate(zip(encoders, ["0.6", "tab:green"])):
        bars = ax.bar(x + (k - 0.5) * w, vals[e], w, color=col, label=e)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC (held-out patients)")
    ax.set_title("Frozen probe: SSL vs random-encoder floor")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "auroc.png")


def fig_loss(out, ckpt):
    """Loss curves (inv/var/cov vs epoch) from the metrics.csv next to the ckpt."""
    csv_path = os.path.join(os.path.dirname(ckpt), "metrics.csv")
    if not os.path.exists(csv_path):
        print("[viz] no metrics.csv next to ckpt — skip loss curves"); return
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col, lab in [("invariance_loss", "invariance"), ("var_loss", "variance (anti-collapse)"),
                     ("cov_loss", "covariance")]:
        ax.plot(d["epoch"], d[col], label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss term")
    ax.set_title("VICReg sub-losses — variance descending = no collapse")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "loss_curves.png")


def _save(fig, out, name):
    p = os.path.join(out, name); fig.savefig(p, dpi=130); plt.close(fig)
    print("[viz] wrote", p, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--what", default="all",
                    choices=["all", "signals", "psd", "bandpower", "latent", "lda",
                             "scores", "auroc", "loss"])
    ap.add_argument("--out", default="eeg_figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    sfreq = int(cfg.data.sfreq)

    raw = {"signals": fig_signals, "psd": fig_psd, "bandpower": fig_bandpower}
    emb = {"latent": fig_latent, "lda": fig_lda, "scores": fig_scores, "auroc": fig_auroc}

    for k, fn in raw.items():
        if args.what in ("all", k):
            fn(args.out, sfreq)
    if args.what in ("all", "loss"):
        fig_loss(args.out, args.ckpt)
    if args.what == "all" or args.what in emb:
        cache = _feature_cache(cfg, state, device)
        for k, fn in emb.items():
            if args.what in ("all", k):
                fn(args.out, cache)
    print(f"[viz] done -> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
