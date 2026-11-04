"""Neural Network experiments (sklearn MLP + PyTorch MLP) for Covertype.

SGD-only constraint: no momentum, no Adam/RMSprop.
Extra credit: activation function study (ReLU, GELU, SiLU, Tanh) on PyTorch.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    RANDOM_SEED, N_CV, SCORING,
    ensure_dirs, save_fig, save_json, eval_metrics,
    compute_learning_curve, plot_learning_curve,
    plot_confusion_matrix, plot_per_class_bar,
    print_per_class_metrics, print_section,
)

# sklearn MLP search grids
ARCH_GRID  = [(100,), (200,), (150, 75), (100, 50), (200, 100)]
LR_GRID    = [0.001, 0.01, 0.05, 0.1]
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1]
WIDTH_GRID = [25, 50, 100, 150, 200, 256]

# Architectures for depth-vs-width study (~10-12k params each with 54 inputs, 7 outputs)
#   (200,)      : 54×200 + 200×7              ≈ 12 200 params  [1 layer]
#   (100, 50)   : 54×100 + 100×50 + 50×7     ≈ 10 750 params  [2 layers]
#   (64, 64, 32): 54×64 + 64×64 + 64×32 + 32×7 ≈  9 800 params  [3 layers]
DEPTH_WIDTH_ARCHS = [(200,), (100, 50), (64, 64, 32)]
DEPTH_WIDTH_LABELS = ["1L-200", "2L-100×50", "3L-64×64×32"]

# PyTorch fixed params
PT_HIDDEN      = 150
PT_LR          = 0.05
PT_WD          = 1e-4
PT_BATCH       = 128
PT_EPOCHS      = 500
PT_PATIENCE    = 30
ACT_EPOCHS     = 300   # max epochs for the activation function study (extra credit)

# PyTorch LR candidates to search
PT_LR_GRID = [0.01, 0.05, 0.1]


def _cv_mean(est, X, y) -> float:
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_SEED)
    return cross_val_score(est, X, y, cv=cv, scoring=SCORING, n_jobs=-1).mean()


def _cv_both(est, X, y):
    """Return (mean_val_score, mean_train_score) in one CV pass."""
    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_SEED)
    res = cross_validate(est, X, y, cv=cv, scoring=SCORING,
                         return_train_score=True, n_jobs=-1)
    return res["test_score"].mean(), res["train_score"].mean()


# ── Sklearn MLP ───────────────────────────────────────────────────────────────
def run_nn_sklearn(X_train, y_train, X_test, y_test, tag: str = "covertype") -> dict:
    """
    Sequential tuning: architecture → learning_rate_init → alpha.
    SGD solver, no momentum, no Nesterov.
    """
    ensure_dirs()
    print_section(f"NN sklearn  [{tag}]")

    # Scale features: MLPs trained with gradient descent are sensitive to feature
    # magnitude. Fit scaler on training data only to prevent test-set leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("  Features standardised (StandardScaler fit on training set).")

    # Step 1: architecture
    print("  Tuning architecture ...")
    arch_scores = []
    for arch in ARCH_GRID:
        est = MLPClassifier(hidden_layer_sizes=arch, solver="sgd",
                            momentum=0.0, nesterovs_momentum=False,
                            learning_rate_init=0.01,
                            max_iter=200, early_stopping=True, n_iter_no_change=20,
                            random_state=RANDOM_SEED)
        arch_scores.append(_cv_mean(est, X_train, y_train))
        print(f"    arch={arch}  CV={arch_scores[-1]:.4f}")

    best_arch = ARCH_GRID[int(np.argmax(arch_scores))]
    print(f"  Best arch={best_arch}")

    # Step 2: learning rate
    print("  Tuning learning rate ...")
    lr_scores = []
    for lr in LR_GRID:
        est = MLPClassifier(hidden_layer_sizes=best_arch, solver="sgd",
                            momentum=0.0, nesterovs_momentum=False,
                            learning_rate_init=lr,
                            max_iter=200, early_stopping=True, n_iter_no_change=20,
                            random_state=RANDOM_SEED)
        lr_scores.append(_cv_mean(est, X_train, y_train))
        print(f"    lr={lr}  CV={lr_scores[-1]:.4f}")

    best_lr = LR_GRID[int(np.argmax(lr_scores))]
    print(f"  Best lr={best_lr}")

    # Step 3: L2 alpha
    print("  Tuning alpha ...")
    alpha_scores = []
    for a in ALPHA_GRID:
        est = MLPClassifier(hidden_layer_sizes=best_arch, solver="sgd",
                            momentum=0.0, nesterovs_momentum=False,
                            learning_rate_init=best_lr,
                            alpha=a, max_iter=200, early_stopping=True,
                            n_iter_no_change=20, random_state=RANDOM_SEED)
        alpha_scores.append(_cv_mean(est, X_train, y_train))
        print(f"    alpha={a}  CV={alpha_scores[-1]:.4f}")

    best_alpha = ALPHA_GRID[int(np.argmax(alpha_scores))]
    print(f"  Best alpha={best_alpha}")

    # Final model (fit again to access loss_curve_)
    final = MLPClassifier(
        hidden_layer_sizes=best_arch, solver="sgd",
        momentum=0.0, nesterovs_momentum=False,
        learning_rate_init=best_lr, alpha=best_alpha,
        max_iter=500, early_stopping=True, n_iter_no_change=20,
        validation_fraction=0.1, random_state=RANDOM_SEED,
    )
    t0 = time.perf_counter()
    final.fit(X_train, y_train)
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = final.predict(X_test)
    pred_t = time.perf_counter() - t0

    metrics = eval_metrics(y_test, y_pred, fit_t, pred_t)
    metrics.update({"arch": str(best_arch), "lr": best_lr,
                    "alpha": best_alpha, "n_epochs": final.n_iter_})
    print(f"  Test Macro F1={metrics['macro_f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"Epochs={final.n_iter_}  Fit={fit_t:.2f}s")
    print(f"  Hyperparams: solver=SGD, momentum=0, lr={best_lr}, alpha={best_alpha}, "
          f"early_stopping=True, n_iter_no_change=20, validation_fraction=0.1")

    # Figure 1: Epoch loss curve (dual y-axis: train loss left, val F1 right)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(final.loss_curve_, color="#4C72B0", label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss", color="#4C72B0")
    ax.tick_params(axis="y", colors="#4C72B0")
    ax.set_title(f"NN (sklearn SGD) — Epoch Curve [{tag}]")
    if hasattr(final, "validation_scores_") and final.validation_scores_ is not None:
        ax2 = ax.twinx()
        ax2.plot(final.validation_scores_, color="#DD8452", ls="--", label="Val Macro F1")
        ax2.set_ylabel("Val Macro F1", color="#DD8452")
        ax2.tick_params(axis="y", colors="#DD8452")
        # combine handles from both axes so both labels appear in legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    save_fig(fig, f"nn_sklearn_{tag}_epoch.png")

    # Figure 2: Width complexity
    print("  Width complexity study ...")
    width_val = []   # validation (CV)
    width_trn = []   # training
    for w in WIDTH_GRID:
        est = MLPClassifier(hidden_layer_sizes=(w,), solver="sgd",
                            momentum=0.0, nesterovs_momentum=False,
                            learning_rate_init=best_lr, alpha=best_alpha,
                            max_iter=300, early_stopping=True, n_iter_no_change=20,
                            random_state=RANDOM_SEED)
        val_s, trn_s = _cv_both(est, X_train, y_train)
        width_val.append(val_s)
        width_trn.append(trn_s)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(WIDTH_GRID, width_trn, "s--", color="#55A868", ms=6, label="Train")
    ax.plot(WIDTH_GRID, width_val, "o-",  color="#4C72B0", ms=6, label="Validation (CV)")
    ax.set_xlabel("Hidden layer width")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"NN (sklearn) — Width Complexity [{tag}]")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, f"nn_sklearn_{tag}_width.png")

    # Figure 3: Depth-vs-width study (parameter-matched architectures)
    print("  Depth-vs-width study ...")
    dw_scores = []
    for arch in DEPTH_WIDTH_ARCHS:
        est = MLPClassifier(hidden_layer_sizes=arch, solver="sgd",
                            momentum=0.0, nesterovs_momentum=False,
                            learning_rate_init=best_lr, alpha=best_alpha,
                            max_iter=300, early_stopping=True, n_iter_no_change=20,
                            random_state=RANDOM_SEED)
        dw_scores.append(_cv_mean(est, X_train, y_train))
        print(f"    arch={arch}  CV={dw_scores[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(DEPTH_WIDTH_LABELS, dw_scores,
                  color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.85)
    ax.bar_label(bars, labels=[f"{s:.4f}" for s in dw_scores],
                 padding=3, fontsize=9)
    ax.set_ylabel("CV Macro F1")
    ax.set_ylim(min(dw_scores) - 0.02, max(dw_scores) + 0.04)
    ax.set_title(f"NN (sklearn) — Depth vs Width (~equal params) [{tag}]")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, f"nn_sklearn_{tag}_depth_vs_width.png")

    # Figure 4: Learning curve
    lc_est = MLPClassifier(
        hidden_layer_sizes=best_arch, solver="sgd",
        momentum=0.0, nesterovs_momentum=False,
        learning_rate_init=best_lr, alpha=best_alpha,
        max_iter=300, early_stopping=True, n_iter_no_change=20,
        random_state=RANDOM_SEED,
    )
    sizes, tr_sc, val_sc = compute_learning_curve(lc_est, X_train, y_train, n_points=7)
    plot_learning_curve(sizes, tr_sc, val_sc,
                        f"NN (sklearn SGD) — Learning Curve [{tag}]",
                        f"nn_sklearn_{tag}_lc.png")

    # Figure 5: Confusion matrix
    plot_confusion_matrix(y_test, y_pred,
                          f"NN (sklearn SGD) — Confusion Matrix [{tag}]",
                          f"nn_sklearn_{tag}_cm.png")

    # Figure 6: Per-class F1
    plot_per_class_bar(y_test, y_pred,
                       f"NN (sklearn) — Per-Class F1 [{tag}]",
                       f"nn_sklearn_{tag}_per_class_f1.png")
    print_per_class_metrics(y_test, y_pred, f"NN sklearn [{tag}]")

    save_json(metrics, f"nn_sklearn_{tag}_metrics.json")
    return metrics


# ── PyTorch MLP ───────────────────────────────────────────────────────────────
def run_nn_pytorch(X_train, y_train, X_test, y_test, tag: str = "covertype") -> dict:
    """
    Fixed-hyperparameter PyTorch MLP with SGD (no momentum).
    Includes: width complexity, depth-vs-width study, activation function study.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("  [PyTorch not available — skipping]")
        return {"macro_f1": None, "accuracy": None, "fit_s": None, "predict_s": None}

    ensure_dirs()
    print_section(f"NN PyTorch  [{tag}]")

    # Scale features before any splitting — scaler is fit on the full X_train.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("  Features standardised (StandardScaler fit on training set).")
    print(f"  Hyperparams: SGD lr={PT_LR}, weight_decay={PT_WD}, "
          f"batch={PT_BATCH}, max_epochs={PT_EPOCHS}, patience={PT_PATIENCE}, "
          f"momentum=0 (pure SGD).")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    n_feats = X_train.shape[1]
    classes = np.unique(y_train)
    n_cls   = len(classes)
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    inv_map   = {i: c for c, i in label_map.items()}

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.array(sorted(classes)), y=y_train)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    def make_loader(X, y, shuffle: bool = True):
        X_t = torch.tensor(X.astype(np.float32))
        y_t = torch.tensor(np.array([label_map[yi] for yi in y], dtype=np.int64))
        return DataLoader(TensorDataset(X_t, y_t), batch_size=PT_BATCH, shuffle=shuffle)

    class MLP(nn.Module):
        """Variable-depth MLP. `hidden` may be an int (1 layer) or a tuple of ints."""
        def __init__(self, n_in, hidden, n_out, act_cls=nn.ReLU):
            super().__init__()
            sizes = [n_in] + (list(hidden) if hasattr(hidden, "__iter__") else [hidden])
            layers = []
            for i in range(len(sizes) - 1):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), act_cls()]
            layers.append(nn.Linear(sizes[-1], n_out))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def train_model(model, loader_tr, loader_val, epochs, lr, wd, patience):
        model = model.to(device)
        crit  = nn.CrossEntropyLoss(weight=cw_tensor)
        opt   = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.0)
        best_val_f1, patience_ctr, best_state = -1.0, 0, None
        history = {"loss": [], "val_f1": []}

        for ep in range(epochs):
            model.train()
            ep_loss = 0.0
            for xb, yb in loader_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(xb)
            history["loss"].append(ep_loss / len(loader_tr.dataset))

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in loader_val:
                    preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                    trues.extend(yb.numpy())
            val_f1 = f1_score(trues, preds, average="macro", zero_division=0)
            history["val_f1"].append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                patience_ctr = 0
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model, history, ep + 1

    def predict_all(model, X):
        model.eval()
        X_t = torch.tensor(X.astype(np.float32)).to(device)
        with torch.no_grad():
            out = model(X_t).argmax(1).cpu().numpy()
        return np.array([inv_map[i] for i in out])

    # 10% stratified validation split
    from sklearn.model_selection import train_test_split as tts
    X_tr2, X_val, y_tr2, y_val = tts(
        X_train, y_train, test_size=0.1,
        stratify=y_train, random_state=RANDOM_SEED,
    )
    loader_tr  = make_loader(X_tr2, y_tr2)
    loader_val = make_loader(X_val, y_val, shuffle=False)

    # ── LR tuning ─────────────────────────────────────────────────────────
    print("  Tuning PyTorch LR ...")
    best_pt_lr, best_pt_lr_f1 = PT_LR, -1.0
    for lr_cand in PT_LR_GRID:
        m_cand = MLP(n_feats, PT_HIDDEN, n_cls)
        m_cand, _, _ = train_model(m_cand, loader_tr, loader_val, 100, lr_cand, PT_WD, 15)
        cand_preds = predict_all(m_cand, X_val)
        cand_f1 = f1_score(y_val, cand_preds, average="macro", zero_division=0)
        print(f"    lr={lr_cand}  val Macro F1={cand_f1:.4f}")
        if cand_f1 > best_pt_lr_f1:
            best_pt_lr_f1 = cand_f1
            best_pt_lr = lr_cand
    print(f"  Best PyTorch LR={best_pt_lr}")

    # ── Final model ───────────────────────────────────────────────────────
    print(f"  Training final model (h={PT_HIDDEN}, lr={best_pt_lr}, wd={PT_WD}) ...")
    model = MLP(n_feats, PT_HIDDEN, n_cls)
    t0    = time.perf_counter()
    model, history, stop_ep = train_model(
        model, loader_tr, loader_val, PT_EPOCHS, best_pt_lr, PT_WD, PT_PATIENCE
    )
    fit_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = predict_all(model, X_test)
    pred_t = time.perf_counter() - t0

    metrics = eval_metrics(y_test, y_pred, fit_t, pred_t)
    metrics.update({"hidden": PT_HIDDEN, "lr": best_pt_lr, "wd": PT_WD,
                    "stop_epoch": stop_ep})
    print(f"  Test Macro F1={metrics['macro_f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"Epochs={stop_ep}  Fit={fit_t:.2f}s")

    # Figure 1: Epoch curve (dual y-axis: train loss left, val F1 right)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(history["loss"], color="#4C72B0", label="Train loss")
    ax1.axvline(stop_ep - 1, color="green", ls="--", lw=1.2, label=f"Early stop (ep={stop_ep})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="#4C72B0")
    ax1.tick_params(axis="y", colors="#4C72B0")
    ax1.set_title(f"PyTorch NN — Epoch Curve [{tag}]")
    ax2 = ax1.twinx()
    ax2.plot(history["val_f1"], color="#DD8452", ls="--", label="Val Macro F1")
    ax2.set_ylabel("Val Macro F1", color="#DD8452")
    ax2.tick_params(axis="y", colors="#DD8452")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"nn_pytorch_{tag}_epoch.png")

    # Figure 2: Width complexity (50 epochs per candidate)
    print("  Width complexity study ...")
    width_val_f1 = []   # validation F1
    width_trn_f1 = []   # training F1
    for w in WIDTH_GRID:
        m = MLP(n_feats, w, n_cls)
        m, _, _ = train_model(m, loader_tr, loader_val, 50, PT_LR, PT_WD, 10)
        val_preds = predict_all(m, X_val)
        trn_preds = predict_all(m, X_tr2)
        width_val_f1.append(f1_score(y_val,  val_preds, average="macro", zero_division=0))
        width_trn_f1.append(f1_score(y_tr2,  trn_preds, average="macro", zero_division=0))
        print(f"    w={w}  val Macro F1={width_val_f1[-1]:.4f}  train Macro F1={width_trn_f1[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(WIDTH_GRID, width_trn_f1, "s--", color="#55A868", ms=6, label="Train")
    ax.plot(WIDTH_GRID, width_val_f1, "o-",  color="#DD8452", ms=6, label="Validation")
    ax.set_xlabel("Hidden layer width")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"PyTorch NN — Width Complexity [{tag}]")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, f"nn_pytorch_{tag}_width.png")

    # Figure 3: Depth-vs-width study (parameter-matched architectures, 50 epochs)
    print("  Depth-vs-width study ...")
    dw_f1 = []
    for arch in DEPTH_WIDTH_ARCHS:
        m = MLP(n_feats, arch, n_cls)
        m, _, _ = train_model(m, loader_tr, loader_val, 50, PT_LR, PT_WD, 10)
        preds = predict_all(m, X_val)
        dw_f1.append(f1_score(y_val, preds, average="macro", zero_division=0))
        print(f"    arch={arch}  val Macro F1={dw_f1[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(DEPTH_WIDTH_LABELS, dw_f1,
                  color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.85)
    ax.bar_label(bars, labels=[f"{s:.4f}" for s in dw_f1], padding=3, fontsize=9)
    ax.set_ylabel("Val Macro F1")
    ax.set_ylim(min(dw_f1) - 0.02, max(dw_f1) + 0.04)
    ax.set_title(f"PyTorch NN — Depth vs Width (~equal params) [{tag}]")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, f"nn_pytorch_{tag}_depth_vs_width.png")

    # Figure 4: Learning curve (sklearn proxy, same hypers)
    lc_est = MLPClassifier(
        hidden_layer_sizes=(PT_HIDDEN,), solver="sgd",
        momentum=0.0, nesterovs_momentum=False,
        learning_rate_init=PT_LR, alpha=PT_WD, max_iter=300,
        early_stopping=True, n_iter_no_change=PT_PATIENCE,
        random_state=RANDOM_SEED,
    )
    sizes, tr_sc, val_sc = compute_learning_curve(lc_est, X_train, y_train, n_points=7)
    plot_learning_curve(sizes, tr_sc, val_sc,
                        f"PyTorch NN — Learning Curve [{tag}]",
                        f"nn_pytorch_{tag}_lc.png")

    # Figure 5: Confusion matrix
    plot_confusion_matrix(y_test, y_pred,
                          f"PyTorch NN — Confusion Matrix [{tag}]",
                          f"nn_pytorch_{tag}_cm.png")

    # Figure 6: Per-class F1
    plot_per_class_bar(y_test, y_pred,
                       f"PyTorch NN — Per-Class F1 [{tag}]",
                       f"nn_pytorch_{tag}_per_class_f1.png")
    print_per_class_metrics(y_test, y_pred, f"NN PyTorch [{tag}]")

    # ── Extra credit: activation function study ───────────────────────────
    print("  Activation function study (ReLU / GELU / SiLU / Tanh) ...")
    activations = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "Tanh": nn.Tanh,
    }
    act_colors = {"ReLU": "#4C72B0", "GELU": "#DD8452",
                  "SiLU": "#55A868", "Tanh": "#C44E52"}
    act_results = {}
    fig, ax = plt.subplots(figsize=(9, 5))
    for act_name, act_cls in activations.items():
        m  = MLP(n_feats, PT_HIDDEN, n_cls, act_cls=act_cls)
        m, h, ep = train_model(m, loader_tr, loader_val, ACT_EPOCHS, best_pt_lr, PT_WD, PT_PATIENCE)
        preds = predict_all(m, X_test)
        test_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        act_results[act_name] = {"test_macro_f1": round(test_f1, 4), "stop_epoch": ep}
        ax.plot(h["val_f1"], color=act_colors[act_name],
                label=f"{act_name} (ep={ep}, F1={test_f1:.3f})")
        print(f"    {act_name}: Test Macro F1={test_f1:.4f}  Epochs={ep}")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Macro F1")
    ax.set_title(f"Activation Function Comparison (SGD, no momentum) [{tag}]")
    ax.legend(); ax.grid(alpha=0.3)
    save_fig(fig, f"nn_pytorch_{tag}_activation.png")

    metrics["activation_study"] = act_results
    save_json(metrics, f"nn_pytorch_{tag}_metrics.json")
    return metrics
