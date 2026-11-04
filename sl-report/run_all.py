#!/usr/bin/env python3
"""
run_all.py — Full CS 7641 SL pipeline for Forest Cover Type (Covertype).

Usage:
    python run_all.py [--skip-svm] [--skip-pytorch]

Environment variables:
    COVTYPE_DATA  – path to covtype.data  (default: see sampling.py)
"""

import argparse
import sys
import os
import time
import json

# ── Ensure src/ is on the path ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils import ensure_dirs, FIG_DIR, RES_DIR, SAMPLE_CSV, RANDOM_SEED
from sampling import prepare_splits

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-svm",     action="store_true", help="Skip SVM (slow on large N)")
    p.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch NN")
    return p.parse_args()


def print_summary(all_metrics: dict) -> None:
    print("\n" + "=" * 82)
    print("  FINAL RESULTS SUMMARY — Covertype")
    print("=" * 82)
    fmt = "{:<22} {:>9} {:>9} {:>9} {:>9} {:>9}"
    print(fmt.format("Model", "Macro F1", "Bal.Acc", "Accuracy", "Fit (s)", "Pred (s)"))
    print("-" * 72)

    def _row(name, m):
        if m is None or m.get("macro_f1") is None:
            return
        print(fmt.format(
            name,
            f"{m['macro_f1']:.4f}",
            f"{m.get('balanced_accuracy', 0):.4f}",
            f"{m['accuracy']:.4f}",
            f"{m.get('fit_s', 0):.3f}",
            f"{m.get('predict_s', 0):.4f}",
        ))

    for name, m in all_metrics.items():
        if m is None:
            continue
        if "linear" in m and "rbf" in m:
            _row("SVM (linear)", m["linear"])
            _row("SVM (rbf)",    m["rbf"])
        else:
            _row(name, m)
    print("=" * 82)

    # sklearn vs PyTorch NN comparison (assignment requires explicit comparison)
    sk = all_metrics.get("NN_sklearn")
    pt = all_metrics.get("NN_pytorch")
    if sk and pt and sk.get("macro_f1") and pt.get("macro_f1"):
        print("\n  sklearn MLP vs PyTorch MLP comparison (SGD-only):")
        print(f"    Macro F1 : sklearn={sk['macro_f1']:.4f}  PyTorch={pt['macro_f1']:.4f}  "
              f"Δ={pt['macro_f1'] - sk['macro_f1']:+.4f}")
        print(f"    Bal. Acc : sklearn={sk.get('balanced_accuracy',0):.4f}  "
              f"PyTorch={pt.get('balanced_accuracy',0):.4f}")
        print(f"    Fit time : sklearn={sk.get('fit_s',0):.2f}s  "
              f"PyTorch={pt.get('fit_s',0):.2f}s")
        print(f"    Epochs   : sklearn={sk.get('n_epochs','?')}  "
              f"PyTorch={pt.get('stop_epoch','?')}")
        print("  Interpretation:")
        print("    - sklearn uses its own internal SGD scheduler; PyTorch uses plain")
        print("      optim.SGD with no momentum, giving full control over training.")
        print("    - sklearn's early_stopping uses an internal val split (10%);")
        print("      PyTorch monitors macro-F1 on an explicit stratified val split.")
        print("    - PyTorch is faster when a GPU is available (cuBLAS batched matmul);")
        print("      sklearn is CPU-only via liblinear/BLAS threads.")
        print("    - sklearn is easier to tune via GridSearchCV; PyTorch requires manual")
        print("      hyperparameter loops but offers more architectural flexibility.")
        print("-" * 82)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    print(f"Output figures  → {os.path.abspath(FIG_DIR)}")
    print(f"Output results  → {os.path.abspath(RES_DIR)}")

    # ── Load / create sample, split ──────────────────────────────────────
    X_tr_raw, X_te_raw, y_tr, y_te, X_tr_sc, X_te_sc, _ = prepare_splits(SAMPLE_CSV)

    tag = "covertype"
    all_metrics: dict = {}
    t_total = time.perf_counter()

    # ── Baseline (DummyClassifier — required for leakage check) ──────────
    print("\n[Step 0] DummyClassifier baseline ...")
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(X_tr_raw, y_tr)
    y_dummy = dummy.predict(X_te_raw)
    all_metrics["Baseline (majority)"] = {
        "macro_f1":          round(float(f1_score(y_te, y_dummy, average="macro", zero_division=0)), 4),
        "accuracy":          round(float(accuracy_score(y_te, y_dummy)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_te, y_dummy)), 4),
        "fit_s":   0.0, "predict_s": 0.0,
    }
    print(f"  Majority baseline → Macro F1={all_metrics['Baseline (majority)']['macro_f1']:.4f}  "
          f"Acc={all_metrics['Baseline (majority)']['accuracy']:.4f}  "
          f"Bal.Acc={all_metrics['Baseline (majority)']['balanced_accuracy']:.4f}")

    # ── Decision Tree ─────────────────────────────────────────────────────
    print("\n[Step 1] Decision Tree ...")
    from models.dt import run_dt
    all_metrics["DT"] = run_dt(X_tr_raw, y_tr, X_te_raw, y_te, tag)

    # ── kNN ───────────────────────────────────────────────────────────────
    # Pass raw data: run_knn fits its own StandardScaler internally
    print("\n[Step 2] kNN ...")
    from models.knn import run_knn
    all_metrics["kNN"] = run_knn(X_tr_raw, y_tr, X_te_raw, y_te, tag)

    # ── SVM ───────────────────────────────────────────────────────────────
    # Pass raw data: run_svm fits its own StandardScaler internally
    if not args.skip_svm:
        print("\n[Step 3] SVM ...")
        from models.svm import run_svm
        all_metrics["SVM"] = run_svm(X_tr_raw, y_tr, X_te_raw, y_te, tag)
    else:
        print("\n[Step 3] SVM — SKIPPED")

    # ── NN sklearn ────────────────────────────────────────────────────────
    # Pass raw data: run_nn_sklearn fits its own StandardScaler internally
    print("\n[Step 4] NN (sklearn, SGD) ...")
    from models.nn import run_nn_sklearn
    all_metrics["NN_sklearn"] = run_nn_sklearn(X_tr_raw, y_tr, X_te_raw, y_te, tag)

    # ── NN PyTorch ────────────────────────────────────────────────────────
    # Pass raw data: run_nn_pytorch fits its own StandardScaler internally
    if not args.skip_pytorch:
        print("\n[Step 5] NN (PyTorch, SGD) ...")
        from models.nn import run_nn_pytorch
        all_metrics["NN_pytorch"] = run_nn_pytorch(X_tr_raw, y_tr, X_te_raw, y_te, tag)
    else:
        print("\n[Step 5] NN (PyTorch) — SKIPPED")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    print_summary(all_metrics)

    import platform, sys as _sys
    try:
        import torch as _t
        if _t.cuda.is_available():
            gpu_info = f"GPU={_t.cuda.get_device_name(0)}  CUDA={_t.version.cuda}"
        else:
            gpu_info = "CPU-only (no CUDA)"
    except ImportError:
        gpu_info = "PyTorch not available"
    print(f"\nHardware: {platform.node()} | CPU={platform.processor()} | {gpu_info}")
    print(f"Python {_sys.version.split()[0]} | Total wall-clock time: {elapsed/60:.1f} min")
    print("(Include this hardware note in the report runtime table.)")

    # Save combined results
    out = os.path.join(RES_DIR, "all_metrics.json")
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Combined metrics → {out}")


if __name__ == "__main__":
    main()
