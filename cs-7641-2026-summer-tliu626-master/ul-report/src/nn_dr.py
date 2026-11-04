"""
Part 4 -- Neural network after dimensionality reduction (leakage-safe).

Correct downstream workflow (FAQ §5, §10):
    split first -> fit scaler + DR on TRAIN ONLY -> transform val/test ->
    train the fixed backbone -> tune with val -> evaluate once on held-out test.

The scaler is already train-only inside data.get_split(). Here every DR object
(PCA/ICA/RP) is additionally fit on the scaled TRAIN matrix only and then applied
to val/test, so no test information leaks into the learned representation.

We compare the original-feature baseline against PCA, ICA, and RP features using
Accuracy, Macro-F1, balanced accuracy, per-class F1, and runtime. RP is evaluated
across multiple seeds (its projection is random) so its performance is reported
as a mean +/- spread rather than a single lucky projection.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .model import train_eval_nn


def _fit_transform_dr(kind: str, n_components: int,
                      Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray,
                      seed: int):
    """Fit the DR transform on TRAIN ONLY; apply to all splits. Leakage-safe."""
    from . import dr as DR
    if kind == "original":
        return Xtr, Xva, Xte
    if kind == "pca":
        t = DR.fit_pca(Xtr, n_components, seed)
    elif kind == "ica":
        t = DR.fit_ica(Xtr, n_components, seed)
    elif kind == "rp":
        t = DR.fit_rp(Xtr, n_components, seed)
    else:
        raise ValueError(kind)
    return t.transform(Xtr), t.transform(Xva), t.transform(Xte)


def run_nn_after_dr(components, seeds=None) -> pd.DataFrame:
    """Train the fixed backbone on original + PCA/ICA/RP features.

    ``components`` may be a single int or a dict {"pca":.., "ica":.., "rp":..} of
    per-method justified dims. Returns one row per (representation, seed) with
    metrics + runtime. RP is trained on more seeds (its projection is random), so
    its performance is reported as a distribution, not one lucky projection.
    """
    seeds = seeds or C.SEEDS
    split = D.get_split()
    Xtr, Xva, Xte = split.X_train, split.X_val, split.X_test
    ytr, yva, yte = split.y_train, split.y_val, split.y_test

    if isinstance(components, dict):
        cdim = components
    else:
        cdim = {"pca": int(components), "ica": int(components), "rp": int(components)}

    reps = [
        ("original", Xtr.shape[1], seeds),
        ("pca", cdim["pca"], seeds),
        ("ica", cdim["ica"], seeds),
        ("rp", cdim["rp"], C.RP_SEEDS),   # RP: more seeds for its variability
    ]

    rows = []
    for kind, ncomp, rep_seeds in reps:
        for seed in rep_seeds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Ztr, Zva, Zte = _fit_transform_dr(kind, ncomp, Xtr, Xva, Xte, seed)
            res = train_eval_nn(Ztr, ytr, Zva, yva, Zte, yte, seed=seed)
            row = {"representation": kind, "n_dim": res.in_features, "seed": seed,
                   "fit_time_s": res.fit_time_s, "predict_time_s": res.predict_time_s,
                   "epochs_run": res.epochs_run}
            row.update(res.metrics.summary_row())
            rows.append(row)
            print(f"[nn_dr] {kind:>8s} seed={seed} dim={res.in_features:>2d} "
                  f"acc={res.metrics.accuracy:.3f} "
                  f"macroF1={res.metrics.macro_f1:.3f} "
                  f"balAcc={res.metrics.balanced_accuracy:.3f} "
                  f"fit={res.fit_time_s:.1f}s")
    return pd.DataFrame(rows)


def majority_class_baseline() -> dict:
    """Most-frequent-class test accuracy/macro-F1 -- the reference floor.

    Because the NN uses class-weighted cross-entropy, its balanced accuracy sits
    ABOVE raw accuracy (minority classes are up-weighted); this dummy floor makes
    that trade-off explicit and stops accuracy being read against ~0 chance.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    split = D.get_split()
    dummy = DummyClassifier(strategy="most_frequent").fit(split.X_train, split.y_train)
    yp = dummy.predict(split.X_test)
    return {
        "representation": "majority_class",
        "accuracy": float(accuracy_score(split.y_test, yp)),
        "macro_f1": float(f1_score(split.y_test, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(split.y_test, yp)),
    }


def run_nn_dim_sweep(dims=None, seeds=None) -> pd.DataFrame:
    """NN performance vs n_components for PCA/ICA/RP -- 'how low can we go?'.

    One curve per method answering how many components are enough, which is far
    more convincing than a single-dim comparison. RP additionally exposes its
    seed variability at each dim. Leakage-safe: every transform fit on train only.
    """
    dims = dims or C.NN_DR_SWEEP_DIMS
    seeds = seeds or C.SEEDS
    split = D.get_split()
    Xtr, Xva, Xte = split.X_train, split.X_val, split.X_test
    ytr, yva, yte = split.y_train, split.y_val, split.y_test

    rows = []
    for kind in ("pca", "ica", "rp"):
        for d in dims:
            for seed in seeds:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Ztr, Zva, Zte = _fit_transform_dr(kind, d, Xtr, Xva, Xte, seed)
                res = train_eval_nn(Ztr, ytr, Zva, yva, Zte, yte, seed=seed)
                rows.append({"representation": kind, "n_components": d, "seed": seed,
                             "accuracy": res.metrics.accuracy,
                             "macro_f1": res.metrics.macro_f1,
                             "balanced_accuracy": res.metrics.balanced_accuracy,
                             "fit_time_s": res.fit_time_s})
            sub = [r for r in rows if r["representation"] == kind and r["n_components"] == d]
            m = np.mean([r["macro_f1"] for r in sub])
            print(f"[nn_sweep] {kind:>4s} d={d:>2d} macroF1={m:.3f}")
    return pd.DataFrame(rows)


def summarize_nn(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across seeds per representation for the main-report table."""
    agg = df.groupby("representation").agg(
        n_dim=("n_dim", "first"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        bal_acc_mean=("balanced_accuracy", "mean"),
        bal_acc_std=("balanced_accuracy", "std"),
        fit_time_mean=("fit_time_s", "mean"),
    ).reset_index()
    order = {"original": 0, "pca": 1, "ica": 2, "rp": 3}
    agg["_o"] = agg["representation"].map(order)
    return agg.sort_values("_o").drop(columns="_o").reset_index(drop=True)
