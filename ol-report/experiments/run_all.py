"""
Entry point for the OL Report experiments.

Usage (from repo root):
    python -m ol-report.experiments.run_all baseline
    python -m ol-report.experiments.run_all part1
    python -m ol-report.experiments.run_all part2
    python -m ol-report.experiments.run_all part3
    python -m ol-report.experiments.run_all part4
    python -m ol-report.experiments.run_all all

Suggested order (assignment § "Work Planning"): confirm baseline first, then
run small RO pilots, then Adam ablations, then regularization, then Part 4.
"""

from __future__ import annotations

import sys


def main(stage: str) -> None:
    if stage in ("baseline", "all"):
        from src.baseline import run_baseline
        print("[baseline]"); print(run_baseline())
    if stage in ("part1", "all"):
        from src.part1_ro import run_part1
        print("[part1 RO]"); print(run_part1())
    if stage in ("part2", "all"):
        from src.part2_adam import run_ablations, run_sensitivity
        print("[part2 Adam ablations]"); print(run_ablations())
        run_sensitivity(beta_axis="beta1"); run_sensitivity(beta_axis="beta2")
    if stage in ("part3", "all"):
        from src.part3_reg import run_part3
        print("[part3 regularization]"); print(run_part3())
    if stage in ("part4", "all"):
        from src.part4_integrated import run_part4
        print("[part4 integrated]"); print(run_part4())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
