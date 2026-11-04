"""
Data loading, sampling, and preprocessing for the UL Report.

Two entry points, matching the two leakage regimes in the FAQ (§5 "Do I need
train, validation, and test splits for unsupervised learning?"):

  get_working_sample()  -- EXPLORATORY regime (Parts 1-3: clustering + DR).
      Fit scaling on the whole ~20k working sample. This is allowed because
      labels are used ONLY after fitting, for interpretation/evaluation.

  get_split()           -- DOWNSTREAM regime (Part 4: NN after DR).
      Stratified train/val/test split with the continuous-feature scaler fit on
      TRAIN ONLY, then applied to val/test. Every learned transformation (scaler,
      PCA, ICA, RP) in Part 4 is fit on train only -- the DR objects themselves
      are fit inside nn_dr.py, also on train only.

Preprocessing choice (config.SCALE_ALL): scale continuous cartographic features
only (default), leaving binary wilderness/soil indicators as 0/1.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config as C


# ── Load / sample ─────────────────────────────────────────────────────────────
def load_full() -> pd.DataFrame:
    """Full Covertype: read SL's covtype.data if present, else fetch via sklearn.

    Mirrors the SL Report's loader (sl-report/src/sampling.py) so regenerating the
    stratified sample reproduces the exact SL rows. fetch_covtype downloads the
    UCI Covertype set (~11 MB) to sklearn's data home on first use; CARC compute
    nodes have outbound internet.
    """
    if os.path.exists(C.DATA_PATH):
        return pd.read_csv(C.DATA_PATH, header=None,
                           names=C.FEATURE_COLS + [C.TARGET_COL])

    warnings.warn(
        f"Covertype raw data not found at {C.DATA_PATH}; downloading via "
        "sklearn.datasets.fetch_covtype (same source as the SL loader).",
        stacklevel=2,
    )
    from sklearn.datasets import fetch_covtype

    bunch = fetch_covtype(as_frame=True)
    df = bunch.frame.copy()
    df.columns = C.FEATURE_COLS + [C.TARGET_COL]
    # Cache to the UL outputs dir so subsequent runs skip the download. We do not
    # write into sl-report/ (out of scope; leave the SL checkout untouched).
    raw_cache = os.path.join(C.OUT_DIR, "covtype.data")
    try:
        df.to_csv(raw_cache, header=False, index=False)
        print(f"[data] Cached full Covertype -> {raw_cache}  ({len(df):,} rows)")
    except OSError:
        pass
    return df


def _regenerate_sample() -> pd.DataFrame:
    """Rebuild the SL sample from full Covertype using the SAME procedure/seed.

    The SL Report drew a proportional stratified n=SAMPLE_SIZE sample with
    seed=SAMPLE_SEED (confirmed in the SL run log). Reproducing the identical
    ``train_test_split(full, test_size=SAMPLE_SIZE, stratify=Cover_Type,
    random_state=SAMPLE_SEED)`` call yields the exact same rows SL/OL used, so
    continuity holds. Cached to UL_SAMPLE_PATH so it is regenerated only once.
    """
    if os.path.exists(C.UL_SAMPLE_PATH):
        cached = pd.read_csv(C.UL_SAMPLE_PATH)
        if len(cached) >= 0.8 * C.SAMPLE_SIZE:
            return cached
    full = load_full()
    _, sample = train_test_split(
        full, test_size=C.SAMPLE_SIZE, random_state=C.SAMPLE_SEED,
        stratify=full[C.TARGET_COL],
    )
    sample = sample.reset_index(drop=True)
    sample.to_csv(C.UL_SAMPLE_PATH, index=False)
    print(f"[data] Regenerated SL-faithful sample (n={len(sample):,}, "
          f"seed={C.SAMPLE_SEED}) -> {C.UL_SAMPLE_PATH}")
    return sample


def load_sample() -> pd.DataFrame:
    """Reuse the SL stratified sample; enforce the ~20k assignment requirement.

    Preference order:
      1. The SL committed sample.csv, IF it holds ~SAMPLE_SIZE rows (true reuse).
      2. A UL-cached / freshly regenerated 20k sample built with the SAME
         seed=42 stratified procedure SL used (the committed copy on this
         checkout is a 2k smoke truncation, so we rebuild the full sample).
    """
    if os.path.exists(C.SL_SAMPLE_PATH):
        df = pd.read_csv(C.SL_SAMPLE_PATH)
        if len(df) >= 0.8 * C.SAMPLE_SIZE:
            return df
        warnings.warn(
            f"SL sample at {C.SL_SAMPLE_PATH} has only {len(df)} rows "
            f"(< 0.8*{C.SAMPLE_SIZE}); it is a truncated smoke copy. Regenerating "
            f"the identical stratified n={C.SAMPLE_SIZE}, seed={C.SAMPLE_SEED} "
            "sample from full Covertype to match the SL Report's real sample.",
            stacklevel=2,
        )
    else:
        warnings.warn(
            f"SL sample not found at {C.SL_SAMPLE_PATH}. Rebuilding the stratified "
            f"n={C.SAMPLE_SIZE}, seed={C.SAMPLE_SEED} sample used by SL/OL.",
            stacklevel=2,
        )
    return _regenerate_sample()


def _cont_idx() -> np.ndarray:
    return np.array([C.FEATURE_COLS.index(c) for c in C.CONTINUOUS_COLS])


def _scale_cols(scaler_fit_on: np.ndarray) -> np.ndarray:
    """Which column indices the scaler applies to (all, or continuous only)."""
    if C.SCALE_ALL:
        return np.arange(scaler_fit_on.shape[1])
    return _cont_idx()


# ── Exploratory working sample (Parts 1-3) ────────────────────────────────────
@dataclass
class WorkingSample:
    X: np.ndarray            # preprocessed feature matrix (n, 54)
    y: np.ndarray            # Cover_Type mapped to 0..6 (external label only)
    feature_names: list[str]
    scaler: StandardScaler
    scaled_idx: np.ndarray

    def label_1to7(self) -> np.ndarray:
        return self.y + 1


def get_working_sample() -> WorkingSample:
    """~20k Covertype rows, preprocessed for exploratory clustering/DR.

    Scaling is fit on the whole sample (unsupervised); labels are attached only
    for post-hoc evaluation and are never used to fit anything here.
    """
    df = load_sample()
    X = df[C.FEATURE_COLS].to_numpy(dtype=np.float64)
    y = (df[C.TARGET_COL].to_numpy() - 1).astype(np.int64)  # 1..7 -> 0..6

    idx = _scale_cols(X)
    scaler = StandardScaler().fit(X[:, idx])
    X[:, idx] = scaler.transform(X[:, idx])
    return WorkingSample(X, y, list(C.FEATURE_COLS), scaler, idx)


# ── Leakage-safe split for the downstream NN (Part 4) ─────────────────────────
@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    scaled_idx: np.ndarray

    def sizes(self) -> dict:
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}


def get_split() -> DataSplit:
    """Stratified train/val/test split with scaler fit on TRAIN ONLY.

    Identical strategy to the OL Report so the NN-after-DR comparison stays on the
    same footing as prior reports.
    """
    df = load_sample()
    X = df[C.FEATURE_COLS].to_numpy(dtype=np.float32)
    y = (df[C.TARGET_COL].to_numpy() - 1).astype(np.int64)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=C.TEST_FRAC, random_state=C.SPLIT_SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=C.VAL_FRAC, random_state=C.SPLIT_SEED, stratify=y_tmp,
    )

    idx = _scale_cols(X_train)
    scaler = StandardScaler().fit(X_train[:, idx])
    for arr in (X_train, X_val, X_test):
        arr[:, idx] = scaler.transform(arr[:, idx])

    return DataSplit(X_train, y_train, X_val, y_val, X_test, y_test, scaler, idx)


def class_weights(y: np.ndarray):
    """Inverse-frequency weights for the imbalanced Covertype classes (for NN)."""
    import torch

    counts = np.bincount(y, minlength=C.N_CLASSES).astype(np.float64)
    w = counts.sum() / (C.N_CLASSES * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
