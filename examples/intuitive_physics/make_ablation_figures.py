"""Aggregate ablation runs into jury-ready collapse figures.

Reads each run dir's train_log.csv (always) and probe_results.csv (if present).

Produces in --out:
  fingerprint.png  : pred_loss and std_loss vs epoch, all runs overlaid. The "scissor"
                     signature — healthy runs keep std_loss~0 while pred_loss falls;
                     a collapsed run drives pred_loss->0 while std_loss->1.
  auroc.png        : mean latent AUROC vs epoch per run (needs probe_results.csv).
  summary_bar.png  : final mean latent AUROC per run.

Usage:
  python -m examples.intuitive_physics.make_ablation_figures \
      --runs_dir $EBJEPA_CKPTS/intuitive_physics/ablation [--out figures_ablation]
  # or explicit, ordered:
  python -m examples.intuitive_physics.make_ablation_figures --runs A_seed1 B_seed1 ...
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Preferred display order if these run names are present.
_ORDER = ["full", "weak_std", "no_cov", "no_std", "no_reg"]


def _read_train_log(run_dir: Path):
    path = run_dir / "train_log.csv"
    if not path.exists():
        return None
    cols = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols[k].append(float(v))
    return cols


def _read_probe(run_dir: Path):
    """Return {epoch: mean_latent_auroc} averaged across violation types."""
    path = run_dir / "probe_results.csv"
    if not path.exists():
        return None
    by_epoch = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_epoch[int(float(row["epoch"]))].append(float(row["latent_auroc"]))
    return {ep: sum(v) / len(v) for ep, v in sorted(by_epoch.items())}


# The original validated 50-epoch run lives in a dir named "exp_seed1"; show it as "full".
_LABEL_REMAP = {"exp": "full"}


def _run_label(run_dir: Path) -> str:
    lbl = run_dir.name.replace("_seed1", "")
    return _LABEL_REMAP.get(lbl, lbl)


def _sort_runs(runs):
    def key(r):
        lbl = _run_label(r)
        return (_ORDER.index(lbl) if lbl in _ORDER else len(_ORDER), lbl)
    return sorted(runs, key=key)


def _discover(runs_dir: Path):
    return _sort_runs([d for d in runs_dir.iterdir()
                       if d.is_dir() and (d / "train_log.csv").exists()])


def main():
    args = sys.argv[1:]

    def _get(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    out = Path(_get("--out", "figures_ablation"))
    out.mkdir(parents=True, exist_ok=True)

    if "--runs" in args:
        i = args.index("--runs") + 1
        runs = []
        while i < len(args) and not args[i].startswith("--"):
            runs.append(Path(args[i])); i += 1
        runs = _sort_runs(runs)
    elif "--runs_dir" in args:
        runs = _discover(Path(_get("--runs_dir")))
    else:
        raise SystemExit("provide --runs_dir <dir> or --runs <dir> [<dir> ...]")

    if not runs:
        raise SystemExit("no run dirs with train_log.csv found")
    print("runs:", ", ".join(_run_label(r) for r in runs))

    # --- Figure 1: loss fingerprint (pred_loss + std_loss) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in runs:
        log = _read_train_log(r)
        if not log:
            continue
        lbl = _run_label(r)
        ax1.plot(log["epoch"], log["pred_loss"], label=lbl)
        ax2.plot(log["epoch"], log["std_loss"], label=lbl)
    ax1.set(title="Prediction loss (predcost)", xlabel="epoch", ylabel="pred_loss", yscale="log")
    ax2.set(title="Variance hinge  relu(1 - std)  [1.0 = collapsed]",
            xlabel="epoch", ylabel="std_loss")
    ax2.axhline(1.0, ls="--", c="k", lw=0.8, alpha=0.5)
    for ax in (ax1, ax2):
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fingerprint.png", dpi=140); plt.close(fig)
    print(f"wrote {out/'fingerprint.png'}")

    # --- Figure 2: AUROC vs epoch (if probed) ---
    have_probe = {r: _read_probe(r) for r in runs}
    if any(have_probe.values()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for r in runs:
            p = have_probe[r]
            if p:
                ax.plot(list(p.keys()), list(p.values()), marker="o", ms=3, label=_run_label(r))
        ax.axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.5, label="chance")
        ax.set(title="VoE detection — mean latent AUROC", xlabel="epoch",
               ylabel="AUROC", ylim=(0.45, 1.02))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "auroc.png", dpi=140); plt.close(fig)
        print(f"wrote {out/'auroc.png'}")

    # --- Figure 3: final AUROC bars ---
    labels, final_auroc = [], []
    for r in runs:
        p = have_probe[r]
        labels.append(_run_label(r))
        final_auroc.append(p[max(p)] if p else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#3b7dd8" if a >= 0.8 else "#d8743b" for a in final_auroc]
    ax.bar(range(len(labels)), final_auroc, 0.6, color=colors)
    ax.axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.5)
    ax.set(title="VoE detection at final epoch", ylabel="mean latent AUROC", ylim=(0, 1.05))
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out / "summary_bar.png", dpi=140); plt.close(fig)
    print(f"wrote {out/'summary_bar.png'}")


if __name__ == "__main__":
    main()
