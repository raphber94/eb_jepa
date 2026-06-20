"""
Turn a training run's metrics.csv into a human-readable EVIDENCE bundle:
loss curves, anti-collapse indicator plots, and a pass/fail summary.

Run after training (the SLURM job calls it automatically):
    python -m examples.ac_video_jepa.make_evidence \
        --metrics <run>/model/metrics.csv --out <run>

Produces in --out:
    loss_curves.png          total / pred / reg / probe (+ value if trained) vs step
    collapse_indicators.png  std / cov / sim_t / idm vs step (the 4 anti-collapse terms)
    summary.txt              initial->final per metric + a PASS/WARN collapse verdict

Epoch boundaries are drawn as light vertical lines, with the 1/5/12-epoch snapshot
points (the checkpoints we plan with) labelled.

WHAT TO LOOK FOR:
    pred_loss   : should FALL  (world model learning to predict)
    idm_loss    : should FALL  (latent becoming action-predictive) -- LOAD-BEARING
    std_loss    : LOW & not rising (rising = feature variance dying = collapse)
    cov_loss    : LOW & not rising (rising = features becoming redundant)
    sim_loss_t  : LOW but NONZERO (==0 => temporal collapse)
    value_loss  : (Track 7) should FALL & settle; flat-high/0-variance => value collapse
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on the compute node
import matplotlib.pyplot as plt

SNAP_EPOCHS = [1, 5, 12]  # the checkpoints we later plan with (e-0, e-4, e-11)


def _read_metrics(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    cols = rows[0].keys()
    return {c: [float(r[c]) for r in rows] for c in cols}, len(rows)


def _epoch_boundaries(m):
    """Return {epoch_number: step} for the first step of each epoch (for vlines)."""
    bounds = {}
    for s, e in zip(m["global_step"], m["epoch"]):
        e = int(e)
        if e not in bounds:
            bounds[e] = s
    return bounds


def _mark_epochs(ax, bounds):
    """Light vlines at every epoch start; dashed+labelled at the 1/5/12 snapshots."""
    for e, s in bounds.items():
        ax.axvline(s, color="0.85", lw=0.6, zorder=0)
    for snap in SNAP_EPOCHS:
        if snap in bounds:  # snap epoch N starts at index N (e-(N-1) is "N epochs trained")
            ax.axvline(bounds[snap - 1] if (snap - 1) in bounds else bounds[snap],
                       color="tab:blue", lw=1.0, ls="--", zorder=1)


def _trend(y):
    a, b = (y[1] if len(y) > 1 else y[0]), y[-1]
    rel = (b - a) / (abs(a) + 1e-9)
    return a, b, ("down" if rel < -0.05 else "up" if rel > 0.05 else "flat")


def make_evidence(metrics, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m, n = _read_metrics(metrics)
    step = m["global_step"]
    bounds = _epoch_boundaries(m)
    has_value = max(m.get("value_loss", [0])) > 0  # Track 7 value head trained?

    # ---- 1. LOSS CURVES -----------------------------------------------------
    panels = [
        ("total_loss", "Total loss"),
        ("pred_loss", "Prediction loss (world model)"),
        ("reg_loss", "Regularizer loss (weighted sum)"),
        ("probe_loss", "Probe XY-MSE (position decodability)"),
    ]
    if has_value:
        panels.append(("value_loss", "Value TD loss (Track 7)"))
    ncol = 3 if len(panels) > 4 else 2
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.5 * nrow), squeeze=False)
    for idx, (key, title) in enumerate(panels):
        ax = axes[idx // ncol][idx % ncol]
        ax.plot(step, m[key], lw=1.5)
        _mark_epochs(ax, bounds)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_yscale("log" if min(m[key]) > 0 else "linear")
        ax.grid(alpha=0.3)
    for j in range(len(panels), nrow * ncol):  # hide unused axes
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Loss curves — should trend DOWN (dashed = 1/5/12-epoch snapshots)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=120)
    plt.close(fig)

    # ---- 2. COLLAPSE INDICATORS --------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    specs = [
        (axes[0, 0], "std_loss", "std_loss  (LOW & not rising; rising=variance dying)"),
        (axes[0, 1], "cov_loss", "cov_loss  (LOW & not rising; rising=redundant feats)"),
        (axes[1, 0], "sim_loss_t", "sim_loss_t  (LOW but NONZERO; ==0 => temporal collapse)"),
        (axes[1, 1], "idm_loss", "idm_loss  (should FALL; LOAD-BEARING action info)"),
    ]
    for ax, key, title in specs:
        ax.plot(step, m[key], lw=1.5, color="tab:red")
        _mark_epochs(ax, bounds)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle("Anti-collapse indicators", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "collapse_indicators.png", dpi=120)
    plt.close(fig)

    # ---- 3. SUMMARY + VERDICT ----------------------------------------------
    lines = [f"EVIDENCE SUMMARY  ({n} logged steps, {metrics})", "=" * 60]
    keys = ["total_loss", "pred_loss", "reg_loss", "probe_loss",
            "std_loss", "cov_loss", "sim_loss_t", "idm_loss"]
    if has_value:
        keys.append("value_loss")
    checks = {}
    for key in keys:
        a, b, d = _trend(m[key])
        checks[key] = (a, b, d)
        lines.append(f"  {key:12s}  {a:9.4f} -> {b:9.4f}   ({d})")

    verdict, notes = "PASS", []
    if checks["pred_loss"][2] != "down":
        verdict = "WARN"; notes.append("pred_loss did not fall")
    if checks["idm_loss"][2] == "up":
        verdict = "WARN"; notes.append("idm_loss rising (action info degrading)")
    if checks["std_loss"][2] == "up":
        verdict = "WARN"; notes.append("std_loss rising (feature variance dying)")
    if checks["cov_loss"][2] == "up":
        verdict = "WARN"; notes.append("cov_loss rising (features becoming redundant)")
    if m["sim_loss_t"][-1] < 1e-5:
        verdict = "WARN"; notes.append("sim_loss_t ~0 (temporal collapse)")
    if has_value:
        # value-collapse heuristic: value loss should fall, not stay pinned high.
        if checks["value_loss"][2] == "up":
            verdict = "WARN"; notes.append("value_loss rising (value head not learning)")

    lines += ["", f"COLLAPSE VERDICT: {verdict}"]
    lines += [f"  - {x}" for x in notes] or ["  - all anti-collapse signals nominal"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[make_evidence] wrote loss_curves.png, collapse_indicators.png, summary.txt -> {out}")


if __name__ == "__main__":
    import fire

    fire.Fire(make_evidence)
