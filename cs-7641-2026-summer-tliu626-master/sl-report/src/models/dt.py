"""Decision Tree experiments for Covertype."""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import f1_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    RANDOM_SEED, N_CV, SCORING, CLASS_NAMES,
    ensure_dirs, save_fig, save_json, eval_metrics,
    compute_learning_curve, plot_learning_curve,
    plot_confusion_matrix, plot_per_class_bar,
    print_per_class_metrics, print_section,
)


def _cv_mean(est, X, y) -> float:
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_SEED)
    return cross_val_score(est, X, y, cv=cv, scoring=SCORING, n_jobs=-1).mean()


def _cv_both(est, X, y):
    """Return (mean_val_score, mean_train_score) in one CV pass."""
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_SEED)
    res = cross_validate(est, X, y, cv=cv, scoring=SCORING,
                         return_train_score=True, n_jobs=-1)
    return res["test_score"].mean(), res["train_score"].mean()


def run_dt(X_train, y_train, X_test, y_test, tag: str = "covertype") -> dict:
    """
    Tune DT via:
      0. criterion (gini vs entropy) via CV
      1. ccp_alpha from the pruning path (post-pruning)
      2. max_depth grid search (2-20) with optimal ccp_alpha locked in
    Produces learning curve, two complexity curves, confusion matrix.

    Regularisation choices
    ----------------------
    * post-pruning via ccp_alpha  – removes subtrees that add little information
      gain, directly controlling model complexity on the training data.
    * max_depth  – caps tree depth to prevent overfitting on noisy minority classes.
    * class_weight='balanced'  – compensates for the strong class imbalance in
      Covertype (ratio ~85:1) so minority classes are not ignored.
    """
    ensure_dirs()
    print_section(f"Decision Tree  [{tag}]")

    # ── Step 0: criterion selection ───────────────────────────────────────
    print("  Selecting criterion (gini vs entropy) ...")
    crit_scores = {}
    for crit in ("gini", "entropy"):
        est = DecisionTreeClassifier(
            criterion=crit, class_weight="balanced", random_state=RANDOM_SEED
        )
        crit_scores[crit] = _cv_mean(est, X_train, y_train)
        print(f"    {crit:8s} CV Macro F1={crit_scores[crit]:.4f}")
    best_criterion = max(crit_scores, key=crit_scores.get)
    print(f"  Chosen criterion: {best_criterion}  "
          f"(higher CV Macro F1 on training fold; "
          f"{'entropy weighs rare splits more heavily' if best_criterion == 'entropy' else 'gini is faster and equally accurate here'})")

    # ── Step 1: ccp_alpha tuning ──────────────────────────────────────────
    print("  Tuning ccp_alpha from pruning path ...")
    base_dt = DecisionTreeClassifier(
        criterion=best_criterion, random_state=RANDOM_SEED, class_weight="balanced"
    )
    path    = base_dt.cost_complexity_pruning_path(X_train, y_train)
    alphas  = path.ccp_alphas

    # sample at most 25 alpha values to keep CV tractable
    if len(alphas) > 25:
        idx    = np.round(np.linspace(0, len(alphas) - 1, 25)).astype(int)
        alphas = alphas[idx]

    alpha_scores = []      # validation (CV) scores — used for selection
    alpha_tr_scores = []   # training scores — for plotting only
    for a in alphas:
        est = DecisionTreeClassifier(
            criterion=best_criterion,
            ccp_alpha=a, class_weight="balanced", random_state=RANDOM_SEED
        )
        val_s, trn_s = _cv_both(est, X_train, y_train)
        alpha_scores.append(val_s)
        alpha_tr_scores.append(trn_s)

    best_alpha_idx = int(np.argmax(alpha_scores))
    best_alpha     = float(alphas[best_alpha_idx])
    print(f"  Best ccp_alpha={best_alpha:.6f}  CV Macro F1={alpha_scores[best_alpha_idx]:.4f}")

    # ── Step 2: max_depth tuning with best alpha ──────────────────────────
    print("  Tuning max_depth ...")
    depths          = list(range(2, 21)) + [None]
    depth_scores    = []      # validation (CV) scores — used for selection
    depth_tr_scores = []      # training scores — for plotting only
    for d in depths:
        est = DecisionTreeClassifier(
            criterion=best_criterion,
            max_depth=d, ccp_alpha=best_alpha,
            class_weight="balanced", random_state=RANDOM_SEED,
        )
        val_s, trn_s = _cv_both(est, X_train, y_train)
        depth_scores.append(val_s)
        depth_tr_scores.append(trn_s)

    best_d_idx  = int(np.argmax(depth_scores))
    best_depth  = depths[best_d_idx]
    print(f"  Best max_depth={best_depth}  CV Macro F1={depth_scores[best_d_idx]:.4f}")

    # ── Final model ───────────────────────────────────────────────────────
    final = DecisionTreeClassifier(
        criterion=best_criterion,
        max_depth=best_depth, ccp_alpha=best_alpha,
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    t0 = time.perf_counter()
    final.fit(X_train, y_train)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = final.predict(X_test)
    pred_t = time.perf_counter() - t0

    metrics = eval_metrics(y_test, y_pred, fit_t, pred_t)
    metrics["best_criterion"] = best_criterion
    metrics["best_depth"]     = best_depth
    metrics["best_alpha"]     = round(best_alpha, 6)
    metrics["n_leaves"]       = int(final.get_n_leaves())
    metrics["actual_depth"]   = int(final.get_depth())
    print(f"  Test Macro F1={metrics['macro_f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"Fit={fit_t:.2f}s  Leaves={metrics['n_leaves']}  Depth={metrics['actual_depth']}")
    print(f"  [Summary] criterion={best_criterion}, ccp_alpha={best_alpha:.6f}, "
          f"max_depth={best_depth} (actual={metrics['actual_depth']}), "
          f"leaves={metrics['n_leaves']}, class_weight=balanced")

    # ── Figure 1: Complexity – ccp_alpha ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, alpha_tr_scores, "s--", color="#55A868", ms=4, label="Train")
    ax.plot(alphas, alpha_scores,    "o-",  color="#4C72B0", ms=4, label="Validation (CV)")
    ax.axvline(best_alpha, color="red", ls="--", lw=1.2, label=f"opt α={best_alpha:.5f}")
    ax.set_xlabel("ccp_alpha")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"DT — ccp_alpha Complexity Curve [{tag}]")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, f"dt_{tag}_ccp_alpha.png")

    # ── Figure 2: Complexity – max_depth ─────────────────────────────────
    depth_labels = [str(d) if d is not None else "None" for d in depths]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(depths)), depth_tr_scores, "s--", color="#55A868", ms=5, label="Train")
    ax.plot(range(len(depths)), depth_scores,    "s-",  color="#DD8452", ms=5, label="Validation (CV)")
    ax.axvline(best_d_idx, color="red", ls="--", lw=1.2,
               label=f"opt depth={best_depth}")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depth_labels, fontsize=7)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"DT — max_depth Complexity Curve [{tag}]")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, f"dt_{tag}_depth.png")

    # ── Figure 3: Learning curve ──────────────────────────────────────────
    lc_est = DecisionTreeClassifier(
        criterion=best_criterion,
        max_depth=best_depth, ccp_alpha=best_alpha,
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    sizes, tr_sc, val_sc = compute_learning_curve(lc_est, X_train, y_train)
    plot_learning_curve(sizes, tr_sc, val_sc,
                        f"DT — Learning Curve [{tag}]",
                        f"dt_{tag}_lc.png")

    # ── Figure 4: Confusion matrix ────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred,
                          f"DT — Confusion Matrix [{tag}]",
                          f"dt_{tag}_cm.png")

    # ── Figure 5: Per-class F1 bar ────────────────────────────────────────
    plot_per_class_bar(y_test, y_pred,
                       f"DT — Per-Class F1 [{tag}]",
                       f"dt_{tag}_per_class_f1.png")
    print_per_class_metrics(y_test, y_pred, f"DT [{tag}]")

    save_json(metrics, f"dt_{tag}_metrics.json")
    return metrics
