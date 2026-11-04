"""
Downstream neural network for Part 4 (NN after DR).

This is the SAME PyTorch backbone finalized in the SL Report and reused in OL:
a single 150-unit ReLU hidden layer (in_features -> 150 -> 7) trained with
class-weighted cross-entropy. The best justified training recipe informed by SL
and OL is Adam (OL Part-2 finding: Adam > plain SGD here) with modest weight
decay (OL Part-3 regularization finding) and early stopping on validation loss.

The ONLY thing that changes across representations is ``in_features`` -- the
input layer is resized to the transformed feature dimension after PCA/ICA/RP, as
the assignment explicitly permits. Everything else (topology, optimizer, budget,
seeds, metrics) is held fixed so differences are attributable to the
representation, not the training setup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .metrics import ClassificationMetrics, compute_metrics


class CovertypeMLP(nn.Module):
    """MLP backbone: in_features -> [hidden...] -> N_CLASSES logits."""

    def __init__(self, in_features: int, n_classes: int = C.N_CLASSES,
                 hidden_sizes=None, dropout_p: float = C.DROPOUT_P):
        super().__init__()
        hidden_sizes = hidden_sizes or C.HIDDEN_SIZES
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(in_features: int) -> CovertypeMLP:
    return CovertypeMLP(in_features=in_features).to(C.DEVICE)


def _class_weights(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=C.N_CLASSES).astype(np.float64)
    w = counts.sum() / (C.N_CLASSES * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=C.DEVICE)


@torch.no_grad()
def _evaluate(model, X, y, criterion) -> tuple[float, np.ndarray]:
    model.eval()
    logits = model(torch.from_numpy(X).to(C.DEVICE))
    loss = criterion(logits, torch.from_numpy(y).to(C.DEVICE)).item()
    pred = logits.argmax(1).cpu().numpy()
    return loss, pred


@dataclass
class NNRunResult:
    metrics: ClassificationMetrics
    fit_time_s: float
    predict_time_s: float
    epochs_run: int
    in_features: int
    seed: int


def train_eval_nn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    *, seed: int = 0,
    epochs: int = C.NN_EPOCHS,
    batch_size: int = C.NN_BATCH_SIZE,
    lr: float = C.NN_LR,
    weight_decay: float = C.NN_WEIGHT_DECAY,
    patience: int | None = C.NN_EARLY_STOP_PATIENCE,
) -> NNRunResult:
    """Train the fixed backbone on one (already-transformed) representation.

    All inputs are float32 numpy arrays whose transforms were fit on TRAIN ONLY
    upstream (leakage-safe). Returns test metrics + runtime accounting.
    """
    from .seeds import set_all_seeds
    set_all_seeds(seed)

    X_train = X_train.astype(np.float32); X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    in_features = X_train.shape[1]

    model = build_model(in_features)
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(y_train) if C.CLASS_WEIGHTED else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(y_train)
    Xt = torch.from_numpy(X_train).to(C.DEVICE)
    yt = torch.from_numpy(y_train).to(C.DEVICE)

    g = torch.Generator().manual_seed(seed)
    best_val, best_state, patience_left = float("inf"), None, patience
    epochs_run = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        epochs_run = epoch + 1

        val_loss, _ = _evaluate(model, X_val, y_val, criterion)
        if patience is not None:
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)
    fit_time = time.perf_counter() - t0

    tp = time.perf_counter()
    test_loss, y_pred = _evaluate(model, X_test, y_test, criterion)
    predict_time = time.perf_counter() - tp

    metrics = compute_metrics(y_test, y_pred, val_loss=test_loss)
    return NNRunResult(metrics, fit_time, predict_time, epochs_run, in_features, seed)
