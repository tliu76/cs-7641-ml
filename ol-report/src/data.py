"""
Data loading, sampling, split, and leakage-safe preprocessing for the OL Report.

SL continuity: reuse the same Covertype sample, split strategy, and preprocessing
as the SL Report. Preprocessing (StandardScaler on continuous features) is fit on
TRAIN ONLY, then applied to val/test. Labels 1..7 are mapped to 0..6 for PyTorch.
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


@dataclass
class DataSplit:
    """Preprocessed arrays + fitted scaler. Convert to tensors via make_loaders()."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    cont_idx: np.ndarray  # column indices scaled

    def sizes(self) -> dict:
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}


# ── Load / sample ─────────────────────────────────────────────────────────────
def load_full() -> pd.DataFrame:
    if not os.path.exists(C.DATA_PATH):
        raise FileNotFoundError(
            f"Covertype data not found at {C.DATA_PATH}. Copy the SL Report data here."
        )
    return pd.read_csv(C.DATA_PATH, header=None, names=C.FEATURE_COLS + [C.TARGET_COL])


def load_sample() -> pd.DataFrame:
    """Load the SL stratified sample if present, else rebuild it (and warn)."""
    if os.path.exists(C.SL_SAMPLE_PATH):
        return pd.read_csv(C.SL_SAMPLE_PATH)

    warnings.warn(
        f"SL sample not found at {C.SL_SAMPLE_PATH}. Rebuilding a stratified "
        f"n={C.SAMPLE_SIZE} sample with seed={C.SAMPLE_SEED}. CONFIRM this matches "
        "the SL Report sample or mark the SL comparison as approximate.",
        stacklevel=2,
    )
    full = load_full()
    _, sample = train_test_split(
        full, test_size=C.SAMPLE_SIZE, random_state=C.SAMPLE_SEED,
        stratify=full[C.TARGET_COL],
    )
    return sample.reset_index(drop=True)


# ── Split + preprocessing ─────────────────────────────────────────────────────
def get_split() -> DataSplit:
    """Stratified train/val/test split with train-only scaling. Deterministic."""
    df = load_sample()
    X = df[C.FEATURE_COLS].to_numpy(dtype=np.float32)
    y = (df[C.TARGET_COL].to_numpy() - 1).astype(np.int64)  # 1..7 -> 0..6

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=C.TEST_FRAC, random_state=C.SPLIT_SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=C.VAL_FRAC, random_state=C.SPLIT_SEED, stratify=y_tmp,
    )

    # Fit scaler on TRAIN continuous columns only, then transform all splits.
    cont_idx = np.array([C.FEATURE_COLS.index(c) for c in C.CONTINUOUS_COLS])
    scaler = StandardScaler().fit(X_train[:, cont_idx])
    for arr in (X_train, X_val, X_test):
        arr[:, cont_idx] = scaler.transform(arr[:, cont_idx])

    return DataSplit(X_train, y_train, X_val, y_val, X_test, y_test, scaler, cont_idx)


# ── Torch loaders ─────────────────────────────────────────────────────────────
def make_loaders(split: DataSplit, batch_size: int = C.BATCH_SIZE, seed: int = 0):
    """Build train/val/test DataLoaders (import torch lazily)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .seeds import seed_worker

    def ds(X, y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    g = torch.Generator().manual_seed(seed)
    train = DataLoader(ds(split.X_train, split.y_train), batch_size=batch_size,
                       shuffle=True, generator=g, worker_init_fn=seed_worker)
    val = DataLoader(ds(split.X_val, split.y_val), batch_size=512, shuffle=False)
    test = DataLoader(ds(split.X_test, split.y_test), batch_size=512, shuffle=False)
    return train, val, test


def class_weights(y: np.ndarray):
    """Optional inverse-frequency weights for the imbalanced Covertype classes."""
    import torch

    counts = np.bincount(y, minlength=C.N_CLASSES).astype(np.float64)
    w = counts.sum() / (C.N_CLASSES * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
