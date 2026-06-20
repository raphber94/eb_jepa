"""Multi-seed figures — aggregates probe_results.csv across seeds and plots robustness.

Usage:
    python -m examples.intuitive_physics.make_figures_multiseed \
        --sweep_dir $EBJEPA_CKPTS/intuitive_physics/sweep_multirun_seeds \
        [--fmt png]

Reads probe_results.csv from each exp_seed* subfolder and generates:
  - ms_fig1_auroc.png        — AUROC vs epoch, one line per seed per violation
  - ms_fig2_dprime.png       — d' vs epoch, mean ± std band across seeds
  - ms_fig3_energy_ratio.png — E(imp)/E(plaus) ratio, mean ± std band
  - ms_fig4_training.png     — pred_loss vs epoch for all seeds (robustness check)
  - ms_fig5_final_bar.png    — AUROC at final epoch, grouped bar per violation × seed
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)

VIOL_COLORS = {"teleport": "#e41a1c", "reversal": "#377eb8", "passthrough": "#4daf4a"}
SEED_MARKERS = ["o", "s", "^"]
SEED_ALPHAS  = [1.0, 0.75, 0.55]
DPI = 180


def _load_sweep(sweep_dir: Path) -> dict[str, pd.DataFrame]:
    """Return {seed_label: probe_df} for every exp_seed* subfolder that has probe_results.csv."""
    result = {}
    for seed_dir in sorted(sweep_dir.glob("exp_seed*")):
        csv = seed_dir / "probe_results.csv"
        if csv.exists():
            label = seed_dir.name  # e.g. "exp_seed1"
            result[label] = pd.read_csv(csv)
        else:
            print(f"  [warn] no probe_results.csv in {seed_dir} — skipping")
    if not result:
        raise FileNotFoundError(
            f"No probe_results.csv found under {sweep_dir}. "
            "Run probe_checkpoints.py for each seed first."
        )
    return result


def _load_train_logs(sweep_dir: Path) -> dict[str, pd.DataFrame]:
    result = {}
    for seed_dir in sorted(sweep_dir.glob("exp_seed*")):
        csv = seed_dir / "train_log.csv"
        if csv.exists():
            result[seed_dir.name] = pd.read_csv(csv)
    return result


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def ms_fig1_auroc(probes: dict, out: Path, fmt: str) -> None:
    """AUROC vs epoch — one line per seed, faceted by violation."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=DPI, sharey=True)
    for ax, viol in zip(axes, ["teleport", "reversal", "passthrough"]):
        for i, (seed_label, df) in enumerate(probes.items()):
            sub = df[df["violation"] == viol].sort_values("epoch")
            seed_num = seed_label.replace("exp_seed", "seed ")
            ax.plot(sub["epoch"], sub["latent_auroc"],
                    color=VIOL_COLORS[viol], marker=SEED_MARKERS[i % 3],
                    alpha=SEED_ALPHAS[i % 3], markersize=4, linewidth=1.5,
                    label=seed_num)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
        ax.set_ylim(0.4, 1.05)
        ax.set_title(viol, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("AUROC")
    fig.suptitle("AUROC vs epoch — 3 seeds (latent VoE)", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"  ✓ {out.name}")


def ms_fig2_dprime(probes: dict, out: Path, fmt: str) -> None:
    """d' vs epoch — mean ± std band across seeds, one panel per violation."""
    if not all("d_prime" in df.columns for df in probes.values()):
        print("  [skip] d_prime column missing — re-run probe_checkpoints with updated code")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=DPI, sharey=True)
    for ax, viol in zip(axes, ["teleport", "reversal", "passthrough"]):
        color = VIOL_COLORS[viol]
        epochs_list, dprime_list = [], []
        for df in probes.values():
            sub = df[df["violation"] == viol].sort_values("epoch")
            epochs_list.append(sub["epoch"].values)
            dprime_list.append(sub["d_prime"].values)
        epochs = epochs_list[0]
        mat = np.array(dprime_list)
        mean, std = mat.mean(0), mat.std(0)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        ax.plot(epochs, mean, color=color, linewidth=2, marker="o", markersize=4, label="mean")
        for i, vals in enumerate(mat):
            ax.plot(epochs, vals, color=color, alpha=0.3, linewidth=0.8,
                    linestyle="--", marker=SEED_MARKERS[i % 3], markersize=3)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(viol, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("d'  (gap / σ_pooled)")
    fig.suptitle("d-prime vs epoch — mean ± std across 3 seeds", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"  ✓ {out.name}")


def ms_fig3_energy_ratio(probes: dict, out: Path, fmt: str) -> None:
    """E(imp)/E(plaus) ratio — mean ± std band across seeds."""
    col = "energy_ratio"
    if not all(col in df.columns for df in probes.values()):
        print("  [skip] energy_ratio column missing — re-run probe_checkpoints with updated code")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=DPI, sharey=True)
    for ax, viol in zip(axes, ["teleport", "reversal", "passthrough"]):
        color = VIOL_COLORS[viol]
        mat = np.array([
            df[df["violation"] == viol].sort_values("epoch")[col].values
            for df in probes.values()
        ])
        epochs = list(probes.values())[0][
            list(probes.values())[0]["violation"] == viol
        ].sort_values("epoch")["epoch"].values
        mean, std = mat.mean(0), mat.std(0)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        ax.plot(epochs, mean, color=color, linewidth=2, marker="o", markersize=4, label="mean")
        for i, vals in enumerate(mat):
            ax.plot(epochs, vals, color=color, alpha=0.3, linewidth=0.8,
                    linestyle="--", marker=SEED_MARKERS[i % 3], markersize=3)
        ax.axhline(1, color="gray", linestyle=":", linewidth=1, label="ratio=1 (no signal)")
        ax.set_title(viol, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("E(impossible) / E(plausible)")
    fig.suptitle("Energy ratio vs epoch — mean ± std across 3 seeds", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"  ✓ {out.name}")


def ms_fig4_training(train_logs: dict, out: Path, fmt: str) -> None:
    """pred_loss and std_loss vs epoch for all seeds — sanity / no-collapse check."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=DPI)
    seed_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i, (seed_label, df) in enumerate(train_logs.items()):
        seed_num = seed_label.replace("exp_seed", "seed ")
        c = seed_colors[i % len(seed_colors)]
        ax1.plot(df["epoch"], df["pred_loss"], color=c, linewidth=1.5, label=seed_num)
        ax2.plot(df["epoch"], df["std_loss"],  color=c, linewidth=1.5, label=seed_num)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("pred_loss")
    ax1.set_title("Prediction loss"); ax1.legend(fontsize=8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("std_loss")
    ax2.set_title("Std loss (anti-collapse)"); ax2.legend(fontsize=8)
    fig.suptitle("Training health across seeds", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"  ✓ {out.name}")


def ms_fig5_final_bar(probes: dict, out: Path, fmt: str) -> None:
    """Grouped bar: AUROC at final epoch per violation × seed — the robustness summary."""
    violations = ["teleport", "reversal", "passthrough"]
    seeds = list(probes.keys())
    n_seeds = len(seeds)
    x = np.arange(len(violations))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 4), dpi=DPI)
    seed_hatches = ["", "//", "xx"]
    seed_colors_bar = ["#e41a1c", "#377eb8", "#4daf4a"]

    for i, (seed_label, df) in enumerate(probes.items()):
        final_epoch = df["epoch"].max()
        final = df[df["epoch"] == final_epoch]
        aurocs = [float(final[final["violation"] == v]["latent_auroc"].values[0])
                  for v in violations]
        offset = (i - n_seeds / 2 + 0.5) * width
        bars = ax.bar(x + offset, aurocs, width,
                      label=seed_label.replace("exp_seed", "seed "),
                      color=seed_colors_bar[i % 3], alpha=0.8,
                      hatch=seed_hatches[i % 3], edgecolor="white")
        for bar, val in zip(bars, aurocs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(violations)
    ax.set_ylim(0.4, 1.06)
    ax.set_ylabel("AUROC (final epoch)")
    ax.set_title("Final AUROC per violation × seed — robustness summary")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out), bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if "--sweep_dir" not in args:
        raise SystemExit(
            "usage: python -m examples.intuitive_physics.make_figures_multiseed "
            "--sweep_dir <path> [--fmt png]"
        )

    def _get(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    sweep_dir = Path(_get("--sweep_dir"))
    fmt       = _get("--fmt", "png")

    probes     = _load_sweep(sweep_dir)
    train_logs = _load_train_logs(sweep_dir)

    print(f"Found {len(probes)} seeds: {list(probes.keys())}")

    out_dir = sweep_dir / "figures_multiseed"
    out_dir.mkdir(exist_ok=True)
    print(f"Writing figures to {out_dir}/  (format: {fmt})\n")

    ms_fig1_auroc(probes,       out_dir / f"ms_fig1_auroc.{fmt}",        fmt)
    ms_fig2_dprime(probes,      out_dir / f"ms_fig2_dprime.{fmt}",       fmt)
    ms_fig3_energy_ratio(probes, out_dir / f"ms_fig3_energy_ratio.{fmt}", fmt)
    if train_logs:
        ms_fig4_training(train_logs, out_dir / f"ms_fig4_training.{fmt}", fmt)
    ms_fig5_final_bar(probes,   out_dir / f"ms_fig5_final_bar.{fmt}",    fmt)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
