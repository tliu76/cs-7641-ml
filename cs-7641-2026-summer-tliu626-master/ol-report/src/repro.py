"""
REPRO capture: dump environment + config so the REPRO_OL_{GTusername}.pdf can be
assembled (package versions, seeds, data paths, split/backbone confirmation,
git commit). Run this after your final experiments; paste the output + a single
git commit hash into the REPRO sheet.
"""

from __future__ import annotations

import json
import platform
import subprocess

from . import config as C


def capture() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seeds": C.SEEDS,
        "sample_seed": C.SAMPLE_SEED,
        "split_seed": C.SPLIT_SEED,
        "sample_size": C.SAMPLE_SIZE,
        "splits": {"test_frac": C.TEST_FRAC, "val_frac": C.VAL_FRAC},
        "batch_size": C.BATCH_SIZE,
        "epochs": C.EPOCHS,
        "hidden_sizes": C.HIDDEN_SIZES,
        "ro_param_cap": C.RO_PARAM_CAP,
        "ro_func_eval_budget": C.RO_FUNC_EVAL_BUDGET,
        "data_path": C.DATA_PATH,
    }
    for pkg in ("torch", "numpy", "pandas", "sklearn", "matplotlib"):
        try:
            info[f"{pkg}_version"] = __import__(pkg).__version__
        except Exception:
            info[f"{pkg}_version"] = "not installed"
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        info["git_commit"] = "unknown"
    return info


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
