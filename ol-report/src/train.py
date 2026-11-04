"""
Shared training / evaluation loop with explicit compute accounting.

Compute-accounting rules (assignment § "Compute accounting and fairness"):
  - one minibatch backward pass == one GRADIENT EVALUATION / optimizer update
  - RO objective calls are FUNCTION EVALUATIONS (see ro_algorithms.py)
  - always record wall-clock time
Evaluation is deterministic: model.eval() so dropout is off and norm layers
behave in eval mode. The test set is used ONLY for final evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .metrics import ClassificationMetrics, compute_metrics


@dataclass
class TrainLog:
    """Per-run trajectory + compute accounting."""
    val_loss: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    grad_evals: list[int] = field(default_factory=list)   # cumulative
    wall_clock: list[float] = field(default_factory=list)  # cumulative seconds
    total_grad_evals: int = 0
    epochs_run: int = 0
    seed: int | None = None
    reached_ell_at: int | None = None    # grad-eval count when val_loss <= ell
    status: str = "ok"                    # "ok" | "diverged" | ">budget"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (mean loss, y_true, y_pred) over a loader in deterministic eval mode."""
    model.eval()
    losses, ys, preds = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
        logits = model(xb)
        losses.append(criterion(logits, yb).item() * len(yb))
        ys.append(yb.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    n = sum(len(y) for y in ys)
    return sum(losses) / n, np.concatenate(ys), np.concatenate(preds)


def final_test_metrics(model: nn.Module, test_loader, criterion) -> ClassificationMetrics:
    """Evaluate on the held-out test set (final only)."""
    loss, y_true, y_pred = evaluate(model, test_loader, criterion)
    return compute_metrics(y_true, y_pred, val_loss=loss)


# ── Training ──────────────────────────────────────────────────────────────────
def make_criterion(label_smoothing: float = 0.0, weight: torch.Tensor | None = None):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight)


def make_criterion_for(split, label_smoothing: float = 0.0):
    """Build the OL loss for a given split.

    The SL PyTorch backbone used *balanced class-weighted* cross-entropy; we reuse
    that here (config.CLASS_WEIGHTED) so the OL baseline reproduces the SL network
    and minority-class behaviour stays comparable. Weights are inverse-frequency on
    the TRAIN split only (deterministic), so val/test cross-entropy is weighted the
    same way across every part -- a fixed, held-constant property of the backbone.
    """
    weight = None
    if C.CLASS_WEIGHTED:
        from .data import class_weights
        weight = class_weights(split.y_train).to(C.DEVICE)
    return make_criterion(label_smoothing=label_smoothing, weight=weight)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    *,
    epochs: int = C.EPOCHS,
    ell: float | None = None,
    early_stop_patience: int | None = None,
    input_noise_std: float = 0.0,
    seed: int | None = None,
    log_every: int = 1,
) -> TrainLog:
    """Generic gradient-based training loop used by Parts 0/2/3/4.

    - counts one gradient evaluation per minibatch backward pass
    - records val loss / grad evals / wall-clock for trajectory plots
    - records the grad-eval count at which val_loss first reaches `ell`
    - optional early stopping (monitor val loss, restore best weights)
    - optional train-only Gaussian input noise (Part 3 regularizer)
    """
    log = TrainLog(seed=seed)
    t0 = time.perf_counter()
    grad_evals = 0
    best_val, best_state, patience_left = float("inf"), None, early_stop_patience

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            if input_noise_std > 0:
                xb = xb + torch.randn_like(xb) * input_noise_std
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            grad_evals += 1                       # one backward == one grad eval
            epoch_losses.append(loss.item())

        if not np.isfinite(np.mean(epoch_losses)):
            log.status = "diverged"
            break

        if epoch % log_every == 0 or epoch == epochs - 1:
            val_loss, _, _ = evaluate(model, val_loader, criterion)
            log.val_loss.append(val_loss)
            log.train_loss.append(float(np.mean(epoch_losses)))
            log.grad_evals.append(grad_evals)
            log.wall_clock.append(time.perf_counter() - t0)

            if ell is not None and log.reached_ell_at is None and val_loss <= ell:
                log.reached_ell_at = grad_evals

            if early_stop_patience is not None:
                if val_loss < best_val - 1e-5:
                    best_val, patience_left = val_loss, early_stop_patience
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        log.status = "early_stopped"
                        break

    if early_stop_patience is not None and best_state is not None:
        model.load_state_dict(best_state)   # restore best weights

    log.total_grad_evals = grad_evals
    log.epochs_run = epoch + 1
    return log
