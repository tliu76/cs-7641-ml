"""SVM experiments (Linear + RBF) for Covertype."""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    RANDOM_SEED, N_CV, SCORING,
    ensure_dirs, save_fig, save_json, eval_metrics,
    compute_learning_curve,
    plot_confusion_matrix, plot_per_class_bar,
    print_per_class_metrics, print_section,
)

C_GRID_COARSE  = [0.001, 0.01, 0.1, 1, 10, 100]
GAMMA_GRID     = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]


def _cv_both(est, X, y):
    """Return (mean_val_score, mean_train_score) in one CV pass."""
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_SEED)
    res = cross_validate(est, X, y, cv=cv, scoring=SCORING,
                         return_train_score=True, n_jobs=-1)
    return res["test_score"].mean(), res["train_score"].mean()


def run_svm(X_train, y_train, X_test, y_test, tag: str = "covertype") -> dict:
    """
    Linear SVM: C grid search using LinearSVC (fast).
    RBF SVM   : coarse-to-fine – C first, then gamma.
    Produces kernel comparison, complexity, learning curve, confusion matrix.
    """
    ensure_dirs()
    print_section(f"SVM  [{tag}]")

    # Scale features: both RBF kernel (distance-based) and LinearSVC (margin
    # width depends on feature magnitude) require standardised inputs.
    # Fit only on training data to prevent leakage into the test set.
    # Validation strategy: 5-fold StratifiedKFold preserves class proportions
    # across folds (important for the imbalanced Covertype classes) while
    # keeping per-fold CV runtime manageable on 20 k samples.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("  Features standardised (StandardScaler fit on training set).")
    print(f"  Validation: {N_CV}-fold StratifiedKFold, scoring='{SCORING}'.")

    # ── Linear SVM ────────────────────────────────────────────────────────
    print("  Linear SVM: C grid search ...")
    lin_scores    = []   # validation (CV) — used for selection
    lin_tr_scores = []   # training — for plotting only
    for c in C_GRID_COARSE:
        est = LinearSVC(C=c, class_weight="balanced",
                        max_iter=20000, random_state=RANDOM_SEED)
        val_s, trn_s = _cv_both(est, X_train, y_train)
        lin_scores.append(val_s)
        lin_tr_scores.append(trn_s)
        print(f"    C={c}  CV Macro F1={lin_scores[-1]:.4f}")

    best_c_lin = C_GRID_COARSE[int(np.argmax(lin_scores))]
    print(f"  Linear best C={best_c_lin}  CV F1={max(lin_scores):.4f}")

    final_lin = LinearSVC(C=best_c_lin, class_weight="balanced",
                          max_iter=20000, random_state=RANDOM_SEED)
    t0 = time.perf_counter()
    final_lin.fit(X_train, y_train)
    fit_t_lin = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_lin = final_lin.predict(X_test)
    pred_t_lin = time.perf_counter() - t0
    metrics_lin = eval_metrics(y_test, y_pred_lin, fit_t_lin, pred_t_lin)
    print(f"  Linear Test Macro F1={metrics_lin['macro_f1']:.4f}  "
          f"Acc={metrics_lin['accuracy']:.4f}  Fit={fit_t_lin:.2f}s")

    # ── RBF SVM: coarse-to-fine ───────────────────────────────────────────
    print("  RBF SVM: C grid (step 1) ...")
    rbf_c_scores    = []   # validation (CV) — used for selection
    rbf_c_tr_scores = []   # training — for plotting only
    for c in C_GRID_COARSE:
        est = SVC(C=c, kernel="rbf", class_weight="balanced",
                  random_state=RANDOM_SEED, cache_size=1000)
        val_s, trn_s = _cv_both(est, X_train, y_train)
        rbf_c_scores.append(val_s)
        rbf_c_tr_scores.append(trn_s)
        print(f"    C={c}  CV Macro F1={rbf_c_scores[-1]:.4f}")

    best_c_rbf = C_GRID_COARSE[int(np.argmax(rbf_c_scores))]
    print(f"  RBF best C={best_c_rbf}")

    print("  RBF SVM: gamma grid (step 2) ...")
    rbf_g_scores    = []   # validation (CV) — used for selection
    rbf_g_tr_scores = []   # training — for plotting only
    for g in GAMMA_GRID:
        est = SVC(C=best_c_rbf, gamma=g, kernel="rbf",
                  class_weight="balanced", random_state=RANDOM_SEED,
                  cache_size=1000)
        val_s, trn_s = _cv_both(est, X_train, y_train)
        rbf_g_scores.append(val_s)
        rbf_g_tr_scores.append(trn_s)
        print(f"    gamma={g}  CV Macro F1={rbf_g_scores[-1]:.4f}")

    best_g_rbf = GAMMA_GRID[int(np.argmax(rbf_g_scores))]
    print(f"  RBF best gamma={best_g_rbf}  CV F1={max(rbf_g_scores):.4f}")

    final_rbf = SVC(C=best_c_rbf, gamma=best_g_rbf, kernel="rbf",
                    class_weight="balanced", random_state=RANDOM_SEED,
                    cache_size=1000)
    t0 = time.perf_counter()
    final_rbf.fit(X_train, y_train)
    fit_t_rbf = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_rbf = final_rbf.predict(X_test)
    pred_t_rbf = time.perf_counter() - t0
    metrics_rbf = eval_metrics(y_test, y_pred_rbf, fit_t_rbf, pred_t_rbf)
    print(f"  RBF Test Macro F1={metrics_rbf['macro_f1']:.4f}  "
          f"Acc={metrics_rbf['accuracy']:.4f}  Fit={fit_t_rbf:.2f}s")

    # ── Figure 1: Kernel comparison (C sweep) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(C_GRID_COARSE, lin_tr_scores,  "o--", label="Linear Train",      color="#4C72B0", alpha=0.6)
    ax.plot(C_GRID_COARSE, lin_scores,     "o-",  label="Linear Validation",  color="#4C72B0")
    ax.plot(C_GRID_COARSE, rbf_c_tr_scores,"s--", label="RBF Train",          color="#DD8452", alpha=0.6)
    ax.plot(C_GRID_COARSE, rbf_c_scores,   "s-",  label="RBF Validation",     color="#DD8452")
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"SVM — Kernel Comparison (C sweep) [{tag}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(fig, f"svm_{tag}_kernel_compare.png")

    # ── Figure 2: RBF gamma complexity ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(GAMMA_GRID, rbf_g_tr_scores, "s--", color="#55A868", ms=6, label="Train")
    ax.plot(GAMMA_GRID, rbf_g_scores,    "o-",  color="#DD8452", ms=6, label="Validation (CV)")
    ax.axvline(best_g_rbf, color="red", ls="--", lw=1.2,
               label=f"opt γ={best_g_rbf}")
    ax.set_xscale("log")
    ax.set_xlabel("gamma (log scale)")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"SVM RBF — Gamma Complexity Curve [{tag}]")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, f"svm_{tag}_rbf_gamma.png")

    # ── Figure 3: Learning curves (both kernels) ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, est, lbl, clr in [
        (axes[0], LinearSVC(C=best_c_lin, class_weight="balanced",
                            max_iter=20000, random_state=RANDOM_SEED), "Linear", "#4C72B0"),
        (axes[1], SVC(C=best_c_rbf, gamma=best_g_rbf, kernel="rbf",
                      class_weight="balanced", random_state=RANDOM_SEED,
                      cache_size=1000), "RBF", "#DD8452"),
    ]:
        sizes, tr, val = compute_learning_curve(est, X_train, y_train, n_points=6)
        ax.plot(sizes, tr,  "o-",  label="Train", color=clr)
        ax.plot(sizes, val, "s--", label="Val",   color=clr, alpha=0.6)
        ax.fill_between(sizes, tr - 0.02, tr + 0.02, alpha=0.1, color=clr)
        ax.set_title(f"SVM {lbl} — Learning Curve [{tag}]")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Macro F1")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"svm_{tag}_lc.png")

    # ── Figure 4: Confusion matrices ──────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred_lin,
                          f"SVM Linear — Confusion Matrix [{tag}]",
                          f"svm_{tag}_cm_linear.png")
    plot_confusion_matrix(y_test, y_pred_rbf,
                          f"SVM RBF — Confusion Matrix [{tag}]",
                          f"svm_{tag}_cm_rbf.png")

    # ── Per-class F1 bars ─────────────────────────────────────────────────
    plot_per_class_bar(y_test, y_pred_lin,
                       f"SVM Linear — Per-Class F1 [{tag}]",
                       f"svm_{tag}_linear_per_class_f1.png")
    plot_per_class_bar(y_test, y_pred_rbf,
                       f"SVM RBF — Per-Class F1 [{tag}]",
                       f"svm_{tag}_rbf_per_class_f1.png")
    print_per_class_metrics(y_test, y_pred_lin, f"SVM Linear [{tag}]")
    print_per_class_metrics(y_test, y_pred_rbf, f"SVM RBF [{tag}]")

    metrics = {
        "linear": {**metrics_lin, "best_C": best_c_lin},
        "rbf":    {**metrics_rbf, "best_C": best_c_rbf, "best_gamma": best_g_rbf},
    }
    save_json(metrics, f"svm_{tag}_metrics.json")
    return metrics
