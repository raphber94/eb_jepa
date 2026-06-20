"""EEG — downstream evaluation (the patient-disjoint abnormality probe).

The feature-extraction harness is provided: per recording, encode N evenly-spaced
10 s windows with the FROZEN encoder and mean-pool them into ONE embedding. What
you implement (`# TODO`) is the probe + metric.

GOLDEN RULE — patient-disjoint split: fit the probe on `train` patients, score on
`eval` patients (no subject overlap). A probe that scores well *within* a subject
but collapses across subjects is measuring identity, not pathology — so the held-
out-patient number is the only one that answers the transferability question.

Run:  python -m examples.eeg.eval --ckpt <.../latest.pth.tar>
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, device):
    """Provided: frozen encoder -> [N_rec, D] recording-level features + labels.

    One embedding per recording: encode its N windows and mean-pool them.
    """
    ds = EEGDataset(EEGConfig(split=split, mode="probe"))
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=16,
                                         pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:          # wins: [B, N, C, T]
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        z = encoder.represent(flat).reshape(B, N, -1).mean(dim=1)  # [B, D]
        z = z.cpu().numpy()
        for k in range(B):
            if bool(ok[k]):                  # drop unreadable recordings
                X.append(z[k]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


@torch.no_grad()
def extract_window_features(encoder, split, device):
    """Per-WINDOW features -> [N_rec*N, D] + labels (each window keeps its
    recording's label). This is the headline protocol (LaBraM/BIOT style):
    every window is scored individually, no per-recording pooling.
    """
    ds = EEGDataset(EEGConfig(split=split, mode="probe"))
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=16,
                                         pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:          # wins: [B, N, C, T]
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        z = encoder.represent(flat).reshape(B, N, -1).cpu().numpy()  # [B, N, D]
        for k in range(B):
            if bool(ok[k]):                  # drop unreadable recordings
                for j in range(N):
                    X.append(z[k, j]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


# --------------------------------------------------------------------------- #
# PROBE + METRIC  — # TODO
# --------------------------------------------------------------------------- #
def probe(Xtr, ytr, Xev, yev):
    """Patient-disjoint linear probe on FROZEN features.

    No leakage: standardize on TRAIN stats only (StandardScaler fit on Xtr),
    fit a balanced LogisticRegression, score on the held-out-patient eval set.
    Returns accuracy / balanced-accuracy / AUROC (normal=0 vs abnormal=1).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 roc_auc_score)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xev_s = scaler.transform(Xtr), scaler.transform(Xev)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(Xtr_s, ytr)

    pred = clf.predict(Xev_s)
    proba = clf.predict_proba(Xev_s)[:, 1]
    return {
        "accuracy": float(accuracy_score(yev, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yev, pred)),
        "auroc": float(roc_auc_score(yev, proba)),
        "n_train": int(len(ytr)),
        "n_eval": int(len(yev)),
    }


def _run_protocol(encoder, device, extractor, name):
    print(f"[eeg-eval] [{name}] extracting TRAIN features...", flush=True)
    Xtr, ytr = extractor(encoder, "train", device)
    print(f"[eeg-eval] [{name}] extracting EVAL features (held-out patients)...", flush=True)
    Xev, yev = extractor(encoder, "eval", device)
    m = probe(Xtr, ytr, Xev, yev)
    print(f"[eeg-eval] [{name}] {m}", flush=True)
    return m


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    # --random: skip loading SSL weights -> random-encoder FLOOR (the control that
    # tells us how much the SSL actually adds over the architecture alone).
    random_floor = "--random" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    if random_floor:
        print("[eeg-eval] RANDOM-ENCODER FLOOR (weights NOT loaded)", flush=True)
    else:
        encoder.load_state_dict(state["encoder"])
    encoder.eval()

    # Headline = per-window (LaBraM/BIOT style); secondary = per-recording (pooled).
    _run_protocol(encoder, device, extract_window_features, "per-window (headline)")
    _run_protocol(encoder, device, extract_features, "per-recording (pooled)")


if __name__ == "__main__":
    main()
