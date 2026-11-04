"""
Data loading, stratified sampling, and train/test split for Covertype.

Returns both raw (DT) and scaled (kNN/SVM/NN) feature matrices.
Scaling is fitted on train only to prevent leakage.
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH   = os.environ.get(
    "COVTYPE_DATA",
    os.path.join(os.path.dirname(__file__), "..", "data", "covtype.data"),
)
SAMPLE_SIZE = 20_000
RANDOM_SEED = 42
TEST_FRAC   = 0.20

CONTINUOUS_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLS       = [f"Soil_Type{i}"       for i in range(1, 41)]
FEATURE_COLS    = CONTINUOUS_COLS + WILDERNESS_COLS + SOIL_COLS
TARGET_COL      = "Cover_Type"


def _load_full_data() -> pd.DataFrame:
    """Load full Covertype data from file, or download via sklearn as fallback."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, header=None, names=FEATURE_COLS + [TARGET_COL])
    print(f"[sampling] {DATA_PATH} not found — downloading via sklearn.datasets ...")
    from sklearn.datasets import fetch_covtype
    bunch = fetch_covtype(as_frame=True)
    df = bunch.frame.copy()
    df.columns = FEATURE_COLS + [TARGET_COL]
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, header=False, index=False)
    print(f"[sampling] Saved full dataset → {DATA_PATH}  ({len(df):,} rows)")
    return df


def load_or_create_sample(sample_csv: str) -> pd.DataFrame:
    if os.path.exists(sample_csv):
        df = pd.read_csv(sample_csv)
        print(f"[sampling] Loaded sample  {sample_csv}  shape={df.shape}")
    else:
        print(f"[sampling] Creating {SAMPLE_SIZE:,}-row sample ...")
        full = _load_full_data()
        _, df = train_test_split(
            full,
            test_size=SAMPLE_SIZE,
            stratify=full[TARGET_COL],
            random_state=RANDOM_SEED,
        )
        df = df.reset_index(drop=True)
        os.makedirs(os.path.dirname(sample_csv), exist_ok=True)
        df.to_csv(sample_csv, index=False)
        print(f"[sampling] Saved sample → {sample_csv}")
    return df


def prepare_splits(sample_csv: str):
    """
    Returns
    -------
    X_train, X_test, y_train, y_test              – raw (DT)
    X_train_sc, X_test_sc                         – StandardScaled continuous cols
    scaler: fitted ColumnTransformer
    feature_names: list[str]
    """
    df      = load_or_create_sample(sample_csv)
    X       = df[FEATURE_COLS].values
    y       = df[TARGET_COL].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_FRAC,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Scale continuous features only; binary cols stay [0,1]
    cont_idx = [FEATURE_COLS.index(c) for c in CONTINUOUS_COLS]
    ct = ColumnTransformer(
        [("scale", StandardScaler(), cont_idx)],
        remainder="passthrough",
    )
    X_train_sc = ct.fit_transform(X_train_raw)
    X_test_sc  = ct.transform(X_test_raw)

    print(f"[sampling] Train={len(y_train):,}  Test={len(y_test):,}  "
          f"Features={X_train_raw.shape[1]}")
    return X_train_raw, X_test_raw, y_train, y_test, X_train_sc, X_test_sc, ct
