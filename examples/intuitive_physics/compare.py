"""EB-JEPA vs SimVP vs ConvLSTM — violation-of-expectation comparison.

Loads trained checkpoints, evaluates all models on the same held-out matched
plausible/impossible pairs, and produces:

  1. auroc_comparison.png    — grouped bar chart (AUROC per violation × model)
  2. score_distributions.png — violin plots of anomaly scores
  3. sample_predictions.png  — pixel-model frame predictions on example pairs
  4. comparison_report.txt   — full numeric summary

Usage (2 models)
----------------
    python -m examples.intuitive_physics.compare \\
        --jepa_ckpt  <.../latest.pth.tar> \\
        --simvp_ckpt <.../latest.pth.tar> \\
        [--out_dir   ./figures] [--n_pairs 200]

Usage (3 models — add --convlstm_ckpt)
---------------------------------------
    python -m examples.intuitive_physics.compare \\
        --jepa_ckpt     <.../latest.pth.tar> \\
        --simvp_ckpt    <.../latest.pth.tar> \\
        --convlstm_ckpt <.../latest.pth.tar> \\
        [--out_dir ./figures] [--n_pairs 200]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score

from eb_jepa.training_utils import load_checkpoint, load_config, setup_device
from examples.intuitive_physics.convlstm import ConvLSTM, build_convlstm
from examples.intuitive_physics.eval import clip_energy as jepa_clip_energy
from examples.intuitive_physics.main import build_jepa
from examples.intuitive_physics.simvp import SimVP, build_simvp
from examples.intuitive_physics.stimuli import VIOLATIONS, build_probe_pairs

# ── palette ─────────────────────────────────────────────────────────────────
C_JEPA     = "#2196F3"   # blue
C_SIMVP    = "#FF9800"   # orange
C_CONVLSTM = "#9C27B0"   # purple
C_PLAUS    = "#4CAF50"   # green
C_IMPOS    = "#F44336"   # red
C_CHANCE   = "#E53935"   # red dashed

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── helpers ──────────────────────────────────────────────────────────────────

def _auroc(e_pla: torch.Tensor, e_imp: torch.Tensor) -> float:
    y = np.r_[np.zeros(len(e_pla)), np.ones(len(e_imp))]
    return float(roc_auc_score(y, np.r_[e_pla.numpy(), e_imp.numpy()]))


def _load_jepa(ckpt_path: str, cfg_path: str, device: torch.device):
    cfg = load_config(cfg_path)
    jepa = build_jepa(cfg, device)
    load_checkpoint(ckpt_path, jepa, device=device)
    jepa.eval()
    return jepa, cfg


def _load_pixel_model(ckpt_path: str, cfg_path: str, device: torch.device,
                      builder, model_class):
    cfg = load_config(cfg_path)
    model = builder(cfg).to(device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model, cfg


# ── evaluation ───────────────────────────────────────────────────────────────

def run_evaluation(models: dict, pairs: dict, nsteps: int,
                   device: torch.device) -> dict:
    """Compute per-clip anomaly scores and AUROC for all models.

    Args:
        models: {"jepa": m, "simvp": m, "convlstm": m}  (convlstm optional)
        pairs:  output of build_probe_pairs()
        nsteps: K-step unroll for EB-JEPA
        device

    Returns:
        results[violation][model_key] = {"e_pla", "e_imp", "auroc", "gap"}
    """
    results = {}
    for v in VIOLATIONS:
        plaus  = pairs[v]["plausible"]
        imposs = pairs[v]["impossible"]

        results[v] = {}
        for key, model in models.items():
            if key == "jepa":
                e_pla = jepa_clip_energy(model, plaus,  nsteps, device)
                e_imp = jepa_clip_energy(model, imposs, nsteps, device)
            else:
                e_pla = model.clip_energy(plaus,  device)
                e_imp = model.clip_energy(imposs, device)

            results[v][key] = {
                "e_pla": e_pla,
                "e_imp": e_imp,
                "auroc": _auroc(e_pla, e_imp),
                "gap":   float(e_imp.mean() - e_pla.mean()),
            }
    return results


# ── Figure 1 — AUROC bar chart ────────────────────────────────────────────────

def plot_auroc_bars(results: dict, model_defs: list, out_dir: Path) -> None:
    """Grouped bar chart: violation × model, with a 'mean' group.

    model_defs: list of (key, label, color)
    """
    violations = list(VIOLATIONS)
    n_models = len(model_defs)
    violations_ext = violations + ["mean"]
    x = np.arange(len(violations_ext))
    total_width = 0.7
    width = total_width / n_models

    fig, ax = plt.subplots(figsize=(max(9, 3 * n_models + 4), 5))

    all_aurocs = {}
    for key, label, color in model_defs:
        aurocs = [results[v][key]["auroc"] for v in violations]
        mean_a = float(np.mean(aurocs))
        aurocs_ext = aurocs + [mean_a]
        all_aurocs[key] = aurocs_ext

        offset = (list(k for k, _, _ in model_defs).index(key) - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, aurocs_ext, width,
                      label=label, color=color, alpha=0.88,
                      edgecolor="white", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(0.5, color=C_CHANCE, linestyle="--", linewidth=1.4, label="Chance (0.50)")
    ax.axvline(len(violations) - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([v.capitalize() for v in violations] + ["Mean"], fontsize=11)
    ax.set_ylabel("AUROC  (↑ better)", fontsize=12)
    ax.set_ylim(0.30, 1.08)

    titles = {
        2: "EB-JEPA (latent) vs SimVP (pixel CNN)",
        3: "EB-JEPA (latent) vs SimVP (pixel CNN) vs ConvLSTM (pixel RNN)",
    }
    ax.set_title(
        f"Violation-of-Expectation Detection — AUROC by violation type\n"
        f"{titles.get(n_models, '')}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()

    path = out_dir / "auroc_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ── Figure 2 — Score distributions ───────────────────────────────────────────

def plot_score_distributions(results: dict, model_defs: list, out_dir: Path) -> None:
    violations = list(VIOLATIONS)
    n_models = len(model_defs)

    fig, axes = plt.subplots(n_models, len(violations),
                             figsize=(14, 3.5 * n_models), sharey=False)
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, (mkey, mlabel, mcol) in enumerate(model_defs):
        for col, v in enumerate(violations):
            ax = axes[row, col]
            e_pla = results[v][mkey]["e_pla"].numpy()
            e_imp = results[v][mkey]["e_imp"].numpy()
            auroc = results[v][mkey]["auroc"]
            gap   = results[v][mkey]["gap"]

            all_e = np.concatenate([e_pla, e_imp])
            mu, sigma = all_e.mean(), all_e.std() + 1e-8
            e_pla_n = (e_pla - mu) / sigma
            e_imp_n = (e_imp - mu) / sigma

            vp = ax.violinplot([e_pla_n, e_imp_n], positions=[0, 1], widths=0.7,
                               showmedians=True, showextrema=False)
            for body, fc in zip(vp["bodies"], [C_PLAUS, C_IMPOS]):
                body.set_facecolor(fc)
                body.set_alpha(0.6)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(2)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Plausible", "Impossible"], fontsize=9)
            ax.set_title(
                f"{v.capitalize()}\nAUROC={auroc:.3f}  gap={gap:+.2e}",
                fontsize=9.5, fontweight="bold",
            )
            if col == 0:
                ax.set_ylabel(f"{mlabel}\nNorm. energy (σ)", fontsize=9)
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.tick_params(labelsize=8)

    plaus_patch = mpatches.Patch(color=C_PLAUS, alpha=0.6, label="Plausible")
    impos_patch = mpatches.Patch(color=C_IMPOS, alpha=0.6, label="Impossible")
    fig.legend(handles=[plaus_patch, impos_patch], loc="upper right",
               fontsize=10, framealpha=0.9)
    fig.suptitle(
        "Anomaly score distributions — plausible vs impossible clips\n"
        "(normalised within each model; wider separation = better detector)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / "score_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ── Figure 3 — Pixel-model frame predictions ──────────────────────────────────

@torch.no_grad()
def plot_sample_predictions(
    pixel_models: list,
    pairs: dict,
    device: torch.device,
    out_dir: Path,
    example_idx: int = 0,
) -> None:
    """For each pixel model and each violation: GT / prediction / |error| rows.

    pixel_models: list of (key, label, model_object)
    """
    violations = list(VIOLATIONS)
    if not pixel_models:
        return

    T_in  = pixel_models[0][2].T_in
    T_out = pixel_models[0][2].T_out
    T = T_in + T_out

    # Layout: for each violation, 2 clips × (1 GT row + N_pixel_models pred rows + N_pixel_models err rows)
    n_pixel = len(pixel_models)
    rows_per_clip = 1 + 2 * n_pixel    # GT + (pred+err) × models
    n_clips_per_viol = 2               # plausible + impossible
    n_rows_total = len(violations) * n_clips_per_viol * rows_per_clip

    fig, axes = plt.subplots(n_rows_total, T,
                             figsize=(T * 1.3, n_rows_total * 1.3))

    for vi, v in enumerate(violations):
        plaus_clip  = pairs[v]["plausible"][example_idx]
        imposs_clip = pairs[v]["impossible"][example_idx]
        t_v = int(pairs[v]["t_v"][example_idx])

        for ci, (clip, clip_label) in enumerate(
                [(plaus_clip, "Plausible"), (imposs_clip, "Impossible")]):
            x_full = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, T, 1, 64, 64]
            ctx = x_full[:, :T_in]
            tgt = x_full[:, T_in:].squeeze(0)                           # [T_out, 1, 64, 64]

            clip_base = (vi * n_clips_per_viol + ci) * rows_per_clip

            # Row 0: ground truth
            for t in range(T):
                ax = axes[clip_base, t]
                ax.imshow(clip[0, t].numpy(), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if t == 0:
                    ax.set_ylabel(
                        f"{v.upper()}  {clip_label}\nGT",
                        fontsize=7, rotation=0, labelpad=52, va="center",
                    )
                if t == t_v and clip_label == "Impossible":
                    for spine in ax.spines.values():
                        spine.set_edgecolor(C_IMPOS)
                        spine.set_linewidth(2)
                        spine.set_visible(True)

            if vi == 0 and ci == 0:
                for t in range(T):
                    axes[0, t].set_title(f"t={t}", fontsize=8)

            # Rows 1..N_pixel: prediction
            # Rows N_pixel+1..2N_pixel: error
            for mi, (mkey, mlabel, pmodel) in enumerate(pixel_models):
                pred = pmodel(ctx).squeeze(0)   # [T_out, 1, 64, 64]
                err  = (pred - tgt).abs()

                pred_row = clip_base + 1 + mi
                err_row  = clip_base + 1 + n_pixel + mi

                for t in range(T):
                    # Prediction row
                    ax_p = axes[pred_row, t]
                    if t < T_in:
                        ax_p.imshow(clip[0, t].numpy(), cmap="gray", vmin=0, vmax=1)
                        ax_p.text(0.5, 0.05, "ctx", transform=ax_p.transAxes,
                                  ha="center", fontsize=6, color="gray")
                    else:
                        ax_p.imshow(pred[t - T_in, 0].cpu().numpy(),
                                    cmap="gray", vmin=0, vmax=1)
                    ax_p.axis("off")
                    if t == 0:
                        ax_p.set_ylabel(f"{mlabel}\npred", fontsize=7,
                                        rotation=0, labelpad=52, va="center")

                    # Error row
                    ax_e = axes[err_row, t]
                    if t < T_in:
                        ax_e.imshow(np.zeros((64, 64)), cmap="hot", vmin=0, vmax=0.5)
                        ax_e.text(0.5, 0.5, "—", transform=ax_e.transAxes,
                                  ha="center", va="center", color="gray", fontsize=8)
                    else:
                        e_frame = err[t - T_in, 0].cpu().numpy()
                        ax_e.imshow(e_frame, cmap="hot", vmin=0, vmax=0.5)
                        ax_e.text(0.98, 0.02, f"{e_frame.mean():.3f}",
                                  transform=ax_e.transAxes, ha="right", va="bottom",
                                  fontsize=6, color="white",
                                  bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
                    ax_e.axis("off")
                    if t == 0:
                        ax_e.set_ylabel(f"{mlabel}\n|err|", fontsize=7,
                                        rotation=0, labelpad=52, va="center")

    model_names = " vs ".join(mlabel for _, mlabel, _ in pixel_models)
    fig.suptitle(
        f"{model_names} — frame predictions on example clips\n"
        "Red border = violation frame  |  |error| coloured hot (dark=low, bright=high)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0.07, 0, 1, 0.97])
    path = out_dir / "sample_predictions.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Summary report ────────────────────────────────────────────────────────────

def print_and_save_report(results: dict, model_defs: list,
                          ckpts: dict, out_dir: Path) -> None:
    """Print and save a full numeric comparison table.

    model_defs: list of (key, label, color)
    ckpts:      {key: ckpt_path_str}
    """
    violations = list(VIOLATIONS)
    col_w = 12

    lines = []
    sep = "=" * (72 + 13 * len(model_defs))
    lines.append(sep)
    lines.append("  Violation-of-Expectation Report — " +
                 " vs ".join(lbl for _, lbl, _ in model_defs))
    lines.append(sep)
    for key, label, _ in model_defs:
        lines.append(f"  {label:20s} checkpoint : {ckpts.get(key, '—')}")
    lines.append("")
    lines.append("  Metric : AUROC  (>0.5 = model surprised by violations; 1.0 = perfect)")
    lines.append("  Gap    : mean(impossible energy) − mean(plausible energy)")
    lines.append("")

    # Header
    header = f"  {'Violation':12s}"
    for _, lbl, _ in model_defs:
        short = lbl.split("(")[0].strip()[:10]
        header += f"  {short + ' AUROC':>{col_w}s}"
    for _, lbl, _ in model_defs:
        short = lbl.split("(")[0].strip()[:10]
        header += f"  {short + ' gap':>{col_w}s}"
    lines.append(header)
    lines.append("  " + "-" * (14 + (col_w + 2) * 2 * len(model_defs)))

    all_aurocs = {key: [] for key, _, _ in model_defs}
    for v in violations:
        row = f"  {v:12s}"
        for key, _, _ in model_defs:
            a = results[v][key]["auroc"]
            all_aurocs[key].append(a)
            row += f"  {a:>{col_w}.4f}"
        for key, _, _ in model_defs:
            g = results[v][key]["gap"]
            row += f"  {g:>+{col_w}.4e}"
        lines.append(row)

    lines.append("  " + "-" * (14 + (col_w + 2) * 2 * len(model_defs)))
    means_row = f"  {'MEAN':12s}"
    means = {}
    for key, _, _ in model_defs:
        m = float(np.mean(all_aurocs[key]))
        means[key] = m
        means_row += f"  {m:>{col_w}.4f}"
    means_row += "  " + "—" * (col_w * len(model_defs))
    lines.append(means_row)

    lines.append("")
    lines.append("  Interpretation:")
    for key, label, _ in model_defs:
        m = means[key]
        tag = " (significantly above chance)" if m > 0.6 else " (at chance)"
        lines.append(f"    {label:30s}: mean AUROC = {m:.4f}{tag}")

    if len(model_defs) > 1:
        best_key = max(means, key=means.get)
        best_lbl = next(lbl for k, lbl, _ in model_defs if k == best_key)
        lines.append(f"\n  Best model: {best_lbl} (mean AUROC {means[best_key]:.4f})")

    lines.append("")
    lines.append("  Per-violation winner:")
    for v in violations:
        best_k = max(model_defs, key=lambda x: results[v][x[0]]["auroc"])[0]
        best_lbl = next(lbl for k, lbl, _ in model_defs if k == best_k)
        best_a = results[v][best_k]["auroc"]
        lines.append(f"    {v:12s}: {best_lbl}  ({best_a:.4f})")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)
    path = out_dir / "comparison_report.txt"
    path.write_text(report)
    print(f"  saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    def _get(flag, default=None):
        if flag in args:
            return args[args.index(flag) + 1]
        return default

    jepa_ckpt     = _get("--jepa_ckpt")
    simvp_ckpt    = _get("--simvp_ckpt")
    convlstm_ckpt = _get("--convlstm_ckpt")

    if jepa_ckpt is None or simvp_ckpt is None:
        raise SystemExit(
            "usage: python -m examples.intuitive_physics.compare "
            "--jepa_ckpt <path> --simvp_ckpt <path> "
            "[--convlstm_ckpt <path>] "
            "[--jepa_cfg <path>] [--simvp_cfg <path>] [--convlstm_cfg <path>] "
            "[--out_dir <dir>] [--n_pairs 200]"
        )

    jepa_cfg_path     = _get("--jepa_cfg",     "examples/intuitive_physics/cfgs/eval.yaml")
    simvp_cfg_path    = _get("--simvp_cfg",    "examples/intuitive_physics/cfgs/simvp_train.yaml")
    convlstm_cfg_path = _get("--convlstm_cfg", "examples/intuitive_physics/cfgs/convlstm_train.yaml")
    out_dir           = Path(_get("--out_dir", "figures"))
    n_pairs           = int(_get("--n_pairs", 200))

    out_dir.mkdir(parents=True, exist_ok=True)
    device = setup_device("auto")

    # ── load models ──
    print("Loading EB-JEPA …")
    jepa, jepa_cfg = _load_jepa(jepa_ckpt, jepa_cfg_path, device)
    print(f"  EB-JEPA    parameters : {sum(p.numel() for p in jepa.parameters()):,}")

    print("Loading SimVP …")
    simvp, _ = _load_pixel_model(simvp_ckpt, simvp_cfg_path, device, build_simvp, SimVP)
    print(f"  SimVP      parameters : {sum(p.numel() for p in simvp.parameters()):,}")

    models = {"jepa": jepa, "simvp": simvp}
    ckpts  = {"jepa": jepa_ckpt, "simvp": simvp_ckpt}

    # ── optional ConvLSTM ──
    convlstm = None
    if convlstm_ckpt is not None:
        print("Loading ConvLSTM …")
        convlstm, _ = _load_pixel_model(
            convlstm_ckpt, convlstm_cfg_path, device, build_convlstm, ConvLSTM)
        print(f"  ConvLSTM   parameters : {sum(p.numel() for p in convlstm.parameters()):,}")
        models["convlstm"] = convlstm
        ckpts["convlstm"]  = convlstm_ckpt

    # model_defs drives all figure / report layouts
    model_defs = [
        ("jepa",  "EB-JEPA (latent)",  C_JEPA),
        ("simvp", "SimVP (pixel CNN)", C_SIMVP),
    ]
    if convlstm is not None:
        model_defs.append(("convlstm", "ConvLSTM (pixel RNN)", C_CONVLSTM))

    # ── build held-out pairs ──
    T = jepa_cfg.data.T
    print(f"\nBuilding {n_pairs} probe pairs per violation (T={T}) …")
    pairs = build_probe_pairs(n_pairs=n_pairs, T=T, seed=999)

    # ── evaluate ──
    print("\nRunning evaluation …")
    nsteps = jepa_cfg.model.steps
    results = run_evaluation(models, pairs, nsteps, device)

    # ── figures ──
    print("\nGenerating figures …")
    plot_auroc_bars(results, model_defs, out_dir)
    plot_score_distributions(results, model_defs, out_dir)

    pixel_models = [(k, lbl, models[k]) for k, lbl, _ in model_defs if k != "jepa"]
    plot_sample_predictions(pixel_models, pairs, device, out_dir)

    # ── report ──
    print()
    print_and_save_report(results, model_defs, ckpts, out_dir)
    print(f"\nAll outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
