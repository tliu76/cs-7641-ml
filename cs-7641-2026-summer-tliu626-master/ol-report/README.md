# OL Report — Optimization & Uncertainty in Learning 

## Layout
```
ol-report/
├── requirements.txt
├── src/
│   ├── config.py          # ALL fixed settings (paths, seeds, budgets, backbone)
│   ├── seeds.py           # seed control (>=3 seeds for main claims)
│   ├── data.py            # SL sample reuse, stratified split, train-only scaling
│   ├── model.py           # PyTorch MLP backbone + freeze/flatten utils for RO
│   ├── metrics.py         # acc, macro-F1, balanced acc, confusion, per-class
│   ├── train.py           # shared loop + compute accounting (grad evals, time)
│   ├── adam_no_bias.py    # custom "Adam w/o bias correction" (Part 2)
│   ├── ro_algorithms.py   # RHC / SA / GA on flat param vector (func-eval budget)
│   ├── plotting.py        # figures + results IO
│   ├── baseline.py        # SL SGD-only baseline (OL reference point)
│   ├── part1_ro.py        # Part 1: RO on final 1-3 layers
│   ├── part2_adam.py      # Part 2: optimizer ablations + sensitivity heatmaps
│   ├── part3_reg.py       # Part 3: regularization study (standard Adam)
│   ├── part4_integrated.py# Part 4 (extra credit): integrated best combination
│   └── repro.py           # REPRO environment/config capture
├── experiments/run_all.py # CLI: baseline | part1 | part2 | part3 | part4 | all
└── outputs/{figures,results,logs}/
```
