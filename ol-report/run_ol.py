"""
Top-level orchestrator for the OL Report experiments.

Run from the ol-report/ directory:
    python run_ol.py all         # baseline -> part1 -> part2 -> part3 -> part4
    python run_ol.py baseline    # individual stages also work
    python run_ol.py part2

Produces, under outputs/{results,figures,logs}/:
  - per-seed trajectories + result CSVs for every part
  - part1 RO best-so-far figure, part2 optimizer-trajectory + sensitivity figures
  - report-only compact figures (baseline comparison, seed stability,
    regularization summary, confusion matrices for baseline vs best combo)
  - repro.json (environment + config + git commit) for the REPRO sheet

Suggested order (assignment "Work Planning"): baseline first, then RO pilots,
then Adam ablations, then regularization, then the integrated Part 4.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Allow "python run_ol.py" from the ol-report/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config as C  # noqa: E402

# The backbone is tiny (54->150->7); spawning one thread per core just adds
# dispatch overhead and thrashes on shared nodes. Cap threads for throughput.
_THREADS = int(os.environ.get("OL_TORCH_THREADS", "4"))
try:
    import torch
    torch.set_num_threads(_THREADS)
except Exception:  # pragma: no cover
    pass


def _banner(msg: str) -> None:
    print("\n" + "=" * 68 + f"\n  {msg}\n" + "=" * 68, flush=True)


def stage_baseline():
    from src.baseline import run_baseline
    _banner("BASELINE  (SL PyTorch SGD-only reference)")
    rows = run_baseline()
    for r in rows:
        print(r, flush=True)


def stage_part1():
    from src.part1_ro import run_part1
    _banner("PART 1  Randomized Optimization on the final layer (RHC/SA/GA)")
    print(run_part1(), flush=True)


def stage_part2():
    from src.part2_adam import run_ablations, run_sensitivity
    _banner("PART 2  Adam-family optimizer ablations")
    print(run_ablations(), flush=True)
    _banner("PART 2  Sensitivity heatmaps  (alpha,beta1) and (alpha,beta2)")
    run_sensitivity(beta_axis="beta1")
    run_sensitivity(beta_axis="beta2")


def stage_part3():
    from src.part3_reg import run_part3
    _banner("PART 3  Regularization study (standard Adam, fixed)")
    print(run_part3(), flush=True)


def stage_part4():
    from src.part4_integrated import run_part4
    _banner("PART 4  Integrated best combination (extra credit)")
    print(run_part4(), flush=True)


def stage_repro():
    from src.repro import capture
    info = capture()
    path = os.path.join(C.RESULT_DIR, "repro.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[repro] wrote {path}", flush=True)


# ── Report-only compact figures (built from saved CSV/JSON) ───────────────────
def stage_report_figures():
    """Compact figures for the main report; robust to missing inputs."""
    _banner("REPORT FIGURES")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.data import get_split, make_loaders
    from src.model import build_model
    from src.plotting import plot_confusion, save_fig
    from src.seeds import set_all_seeds
    from src.train import evaluate, final_test_metrics, make_criterion_for, train_model
    import torch

    R = C.RESULT_DIR

    def _read(name):
        p = os.path.join(R, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    base = _read("baseline_results.csv")
    p2 = _read("part2_ablation_results.csv")
    p3 = _read("part3_regularization_results.csv")

    # (1) Baseline vs best optimizer vs best regularized combo — macro-F1 & bal-acc.
    try:
        groups, f1s, bas = [], [], []
        if base is not None:
            groups.append("SL SGD\n(baseline)")
            f1s.append(base["macro_f1"].mean()); bas.append(base["balanced_accuracy"].mean())
        if p2 is not None:
            best_opt = p2.groupby("optimizer")["macro_f1"].mean().idxmax()
            sub = p2[p2.optimizer == best_opt]
            groups.append(f"best opt\n({best_opt})")
            f1s.append(sub["macro_f1"].mean()); bas.append(sub["balanced_accuracy"].mean())
        if p3 is not None and "best_combo" in set(p3.config):
            sub = p3[p3.config == "best_combo"]
            groups.append("best reg\ncombo")
            f1s.append(sub["macro_f1"].mean()); bas.append(sub["balanced_accuracy"].mean())
        if groups:
            x = np.arange(len(groups)); w = 0.38
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(x - w / 2, f1s, w, label="Macro-F1", color="#4C72B0")
            ax.bar(x + w / 2, bas, w, label="Balanced Acc", color="#DD8452")
            for xi, (a, b) in enumerate(zip(f1s, bas)):
                ax.text(xi - w / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=7)
                ax.text(xi + w / 2, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)
            ax.set_xticks(x, groups); ax.set_ylabel("score")
            ax.set_title("Baseline vs best OL interventions (test)"); ax.legend()
            save_fig(fig, "report_baseline_comparison.png")
    except Exception as e:  # pragma: no cover
        print(f"[fig] baseline comparison skipped: {e}")

    # (2) Part-2 seed stability: mean +/- std macro-F1 per optimizer.
    try:
        if p2 is not None:
            g = p2.groupby("optimizer")["macro_f1"].agg(["mean", "std"]).reindex(
                ["sgd", "sgd_momentum", "nesterov", "adam", "adam_no_bias",
                 "adam_beta1_0", "adamw"])
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.errorbar(range(len(g)), g["mean"], yerr=g["std"].fillna(0),
                        fmt="o", capsize=4, color="#55A868")
            ax.set_xticks(range(len(g)), g.index, rotation=30, ha="right")
            ax.set_ylabel("test Macro-F1"); ax.set_title("Optimizer seed stability (mean $\\pm$ std)")
            save_fig(fig, "report_part2_stability.png")
    except Exception as e:  # pragma: no cover
        print(f"[fig] part2 stability skipped: {e}")

    # (3) Regularization summary: mean val loss & macro-F1 per config.
    try:
        if p3 is not None:
            g = p3.groupby("config").agg(val=("val_loss_final", "mean"),
                                         f1=("macro_f1", "mean")).sort_values("val")
            fig, ax = plt.subplots(figsize=(7.5, 4))
            ax.bar(range(len(g)), g["val"], color="#8172B3")
            ax.set_xticks(range(len(g)), g.index, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("mean validation loss")
            ax.set_title("Regularization study: validation loss by configuration")
            save_fig(fig, "report_part3_regularization.png")
    except Exception as e:  # pragma: no cover
        print(f"[fig] part3 regularization skipped: {e}")

    # (4) Confusion matrices: baseline (SGD) vs best regularized combo, seed 0.
    try:
        split = get_split()
        set_all_seeds(0)
        tr, va, te = make_loaders(split, seed=0)
        crit = make_criterion_for(split)
        m = build_model(dropout_p=0.0)
        opt = torch.optim.SGD([p for p in m.parameters()], lr=C.BASELINE_LR, momentum=0.0)
        train_model(m, tr, va, opt, crit, seed=0)
        mt = final_test_metrics(m, te, crit)
        plot_confusion(mt.confusion, "report_confusion_baseline.png",
                       title="Confusion matrix — SL SGD baseline")

        sel_path = os.path.join(R, "part3_selection.json")
        reg = {}
        if os.path.exists(sel_path):
            reg = json.load(open(sel_path))["best_combo"]["reg"]
        set_all_seeds(0)
        tr, va, te = make_loaders(split, seed=0)
        crit = make_criterion_for(split, label_smoothing=reg.get("label_smoothing", 0.0))
        m = build_model(dropout_p=reg.get("dropout", 0.0))
        opt = torch.optim.Adam([p for p in m.parameters()], lr=1e-3, betas=(0.9, 0.999),
                               weight_decay=reg.get("weight_decay", 0.0))
        train_model(m, tr, va, opt, crit, seed=0,
                    input_noise_std=reg.get("input_noise", 0.0))
        mt = final_test_metrics(m, te, crit)
        plot_confusion(mt.confusion, "report_confusion_best.png",
                       title="Confusion matrix — best regularized Adam")
    except Exception as e:  # pragma: no cover
        print(f"[fig] confusion matrices skipped: {e}")

    print(f"[figures] wrote report figures to {C.FIG_DIR}", flush=True)


STAGES = {
    "baseline": [stage_baseline],
    "part1": [stage_part1],
    "part2": [stage_part2],
    "part3": [stage_part3],
    "part4": [stage_part4],
    "figures": [stage_report_figures],
    "all": [stage_repro, stage_baseline, stage_part1, stage_part2,
            stage_part3, stage_part4, stage_report_figures],
}


def main(stage: str) -> None:
    t0 = time.perf_counter()
    print(f"[run_ol] stage={stage}  seeds={C.SEEDS}  epochs={C.EPOCHS}  "
          f"ro_budget={C.RO_FUNC_EVAL_BUDGET}  device={C.DEVICE}", flush=True)
    for fn in STAGES.get(stage, STAGES["all"]):
        fn()
    print(f"\n[run_ol] stage={stage} finished in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
