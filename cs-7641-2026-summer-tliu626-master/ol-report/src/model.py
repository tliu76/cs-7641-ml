"""
Fixed PyTorch backbone (reused from the SL Report) + utilities for OL.

The architecture MUST match the network you finalized in the SL Report. Only
Part 3 may change regularization internals (dropout); the layer topology stays
fixed across all parts so results are attributable to training interventions.

Also provides:
  - layer freezing for Part 1 (final 1..3 trainable layers under a param cap)
  - flat parameter get/set for randomized optimization (RHC/SA/GA operate on a
    single trainable-parameter vector)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class CovertypeMLP(nn.Module):
    """MLP backbone for 54-feature Covertype -> 7 classes.

    TODO(SL): confirm hidden_sizes/activation match your finalized SL network.
    """

    def __init__(self, in_features: int = C.N_FEATURES, n_classes: int = C.N_CLASSES,
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
        layers.append(nn.Linear(prev, n_classes))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(dropout_p: float = C.DROPOUT_P, hidden_sizes=None) -> CovertypeMLP:
    return CovertypeMLP(hidden_sizes=hidden_sizes, dropout_p=dropout_p).to(C.DEVICE)


# ── Part 1 helpers: freeze all but the final N trainable layers ───────────────
def _linear_layers(model: nn.Module) -> list[nn.Linear]:
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def freeze_to_final_layers(model: nn.Module, n_trainable: int = C.RO_TRAINABLE_LAYERS):
    """Freeze every parameter except those in the final `n_trainable` Linear layers.

    Returns the list of trainable parameters and the trainable-parameter count.
    Raises if the count exceeds RO_PARAM_CAP (assignment hard cap).
    """
    for p in model.parameters():
        p.requires_grad_(False)

    final = _linear_layers(model)[-n_trainable:]
    trainable = []
    for layer in final:
        for p in layer.parameters():
            p.requires_grad_(True)
            trainable.append(p)

    n_params = sum(p.numel() for p in trainable)
    if n_params > C.RO_PARAM_CAP:
        raise ValueError(
            f"Trainable RO params = {n_params:,} > cap {C.RO_PARAM_CAP:,}. "
            "Reduce RO_TRAINABLE_LAYERS or the final-layer width."
        )
    return trainable, n_params


# ── Flat parameter vector get/set for RO ──────────────────────────────────────
def get_flat_params(params) -> torch.Tensor:
    """Concatenate the given parameters into a single detached CPU vector."""
    return torch.cat([p.detach().reshape(-1) for p in params]).cpu()


def set_flat_params(params, flat: torch.Tensor) -> None:
    """Write a flat vector back into the given parameters (in place, no grad)."""
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p).to(p.device))
            offset += n
