"""
Reproducibility metadata for the REPRO PDF: environment, package versions, git
commit, config snapshot, and sample confirmation. Written to
outputs/results/repro.json by run_ul.py.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

from . import config as C


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _pkg_versions() -> dict:
    out = {}
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[pkg] = "not installed"
    return out


def collect(sample_n: int) -> dict:
    return {
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _pkg_versions(),
        "seeds": C.SEEDS,
        "sample_seed": C.SAMPLE_SEED,
        "split_seed": C.SPLIT_SEED,
        "sample_size_config": C.SAMPLE_SIZE,
        "sample_size_actual": sample_n,
        "test_frac": C.TEST_FRAC,
        "val_frac": C.VAL_FRAC,
        "scale_all": C.SCALE_ALL,
        "data_path": C.DATA_PATH,
        "sl_sample_path": C.SL_SAMPLE_PATH,
        "kmeans_ks": C.KMEANS_KS,
        "gmm_components": C.GMM_COMPONENTS,
        "gmm_cov_types": C.GMM_COV_TYPES,
        "pca_var_target": C.PCA_VAR_TARGET,
        "dr_components": C.DR_COMPONENTS,
        "ica_fun": C.ICA_FUN,
        "ica_max_iter": C.ICA_MAX_ITER,
        "rp_kind": C.RP_KIND,
        "nn_recipe": {
            "hidden_sizes": C.HIDDEN_SIZES, "optimizer": C.NN_OPTIMIZER,
            "lr": C.NN_LR, "weight_decay": C.NN_WEIGHT_DECAY,
            "batch_size": C.NN_BATCH_SIZE, "epochs": C.NN_EPOCHS,
            "early_stop_patience": C.NN_EARLY_STOP_PATIENCE,
            "class_weighted": C.CLASS_WEIGHTED,
        },
        "manifold_method": C.MANIFOLD_METHOD,
        "fast_mode": C.FAST,
    }


def write(sample_n: int) -> str:
    path = os.path.join(C.RESULT_DIR, "repro.json")
    with open(path, "w") as f:
        json.dump(collect(sample_n), f, indent=2)
    return path
