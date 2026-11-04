"""kNN experiments for Covertype."""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    RANDOM_SEED, N_CV, SCORING,
    ensure_dirs, save_fig, save_json, eval_metrics,
    compute_learning_curve, plot_learning_curve,
    plot_confusion_matrix, plot_per_class_bar,
    print_per_class_metrics, print_section,
)

K_CANDIDATES      = [1, 3, 5, 7, 11, 15, 21, 31, 41, 51]
WEIGHT_CANDIDATES = ["uniform", "distance"]
METRIC_CANDIDATES = ["euclidean", "manhattan"]


def _cv_mean(est, X, y) -> float:
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_SEED)
    return cross_val_score(est, X, y, cv=cv, scoring=SCORING, n_jobs=-1).mean()


def run_knn(X_train, y_train, X_test, y_test, tag: str = "covertype") -> dict:
    """
    Sequential tuning:
      1. k (odd values 1–51) with uniform/euclidean
      2. weights (uniform vs distance) with best k
      3. metric (euclidean vs manhattan) with best k + weight
    Produces complexity curve (k), learning curve, confusion matrix.
    """
    ensure_dirs()
    print_section(f"kNN  [{tag}]")

    # Scale features: kNN relies on Euclidean/Manhattan distance, so all
    # continuous features must live on the same scale. Fit only on training
    # data to prevent leakage into the test set.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("  Features standardised (StandardScaler fit on training set).")

    # ── Step 1: k grid ────────────────────────────────────────────────────
    print("  Grid search over k ...")
    k_scores = []
    for k in K_CANDIDATES:
        est = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        k_scores.append(_cv_mean(est, X_train, y_train))
        print(f"    k={k:>3}  CV Macro F1={k_scores[-1]:.4f}")

    best_k = K_CANDIDATES[int(np.argmax(k_scores))]
    print(f"  Best k={best_k}")

    # ── Step 2: weighting ─────────────────────────────────────────────────
    w_scores = {}
    for w in WEIGHT_CANDIDATES:
        est = KNeighborsClassifier(n_neighbors=best_k, weights=w, n_jobs=-1)
        w_scores[w] = _cv_mean(est, X_train, y_train)
    best_w = max(w_scores, key=w_scores.get)
    print(f"  Best weights={best_w}  {w_scores}")

    # ── Step 3: distance metric ───────────────────────────────────────────
    m_scores = {}
    for m in METRIC_CANDIDATES:
        est = KNeighborsClassifier(n_neighbors=best_k, weights=best_w,
                                   metric=m, n_jobs=-1)
        m_scores[m] = _cv_mean(est, X_train, y_train)
    best_m = max(m_scores, key=m_scores.get)
    print(f"  Best metric={best_m}  {m_scores}")
    print(f"  Justification: '{best_w}' weighting down-weights distant neighbours; "
          f"'{best_m}' distance selected by CV Macro F1 on standardised features.")

    # ── Final model ───────────────────────────────────────────────────────
    final = KNeighborsClassifier(n_neighbors=best_k, weights=best_w,
                                 metric=best_m, n_jobs=-1)
    t0 = time.perf_counter()
    final.fit(X_train, y_train)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = final.predict(X_test)
    pred_t = time.perf_counter() - t0

    metrics = eval_metrics(y_test, y_pred, fit_t, pred_t)
    metrics["best_k"]      = best_k
    metrics["best_weight"] = best_w
    metrics["best_metric"] = best_m
    print(f"  Test Macro F1={metrics['macro_f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"Fit={fit_t:.3f}s  Pred={pred_t:.3f}s")

    # ── Figure 1: Complexity – k ──────────────────────────────────────────
    # Train score at each k (k=1 gives 1.0 due to self-neighbor — expected overfitting artifact)
    train_k_scores = []
    for k in K_CANDIDATES:
        est = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        est.fit(X_train, y_train)
        train_k_scores.append(f1_score(y_train, est.predict(X_train),
                                       average="macro", zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_CANDIDATES, train_k_scores, "s--", color="#55A868", ms=5,
            alpha=0.7, label="Train")
    ax.plot(K_CANDIDATES, k_scores, "o-", color="#4C72B0", ms=6, label="Validation (CV)")
    ax.axvline(best_k, color="red", ls="--", lw=1.2, label=f"opt k={best_k}")
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"kNN — k Complexity Curve [{tag}]")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, f"knn_{tag}_k.png")

    # ── Figure 2: Learning curve ──────────────────────────────────────────
    lc_est = KNeighborsClassifier(n_neighbors=best_k, weights=best_w,
                                  metric=best_m, n_jobs=-1)
    sizes, tr_sc, val_sc = compute_learning_curve(lc_est, X_train, y_train)
    plot_learning_curve(sizes, tr_sc, val_sc,
                        f"kNN — Learning Curve [{tag}]",
                        f"knn_{tag}_lc.png")

    # ── Figure 3: Confusion matrix ────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred,
                          f"kNN — Confusion Matrix [{tag}]",
                          f"knn_{tag}_cm.png")

    # ── Figure 4: Per-class F1 ────────────────────────────────────────────
    plot_per_class_bar(y_test, y_pred,
                       f"kNN — Per-Class F1 [{tag}]",
                       f"knn_{tag}_per_class_f1.png")
    print_per_class_metrics(y_test, y_pred, f"kNN [{tag}]")

    save_json(metrics, f"knn_{tag}_metrics.json")
    return metrics
