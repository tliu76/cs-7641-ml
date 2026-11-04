"""
Custom "Adam without bias correction" optimizer (required Part 2 ablation).

Standard Adam corrects the zero-initialization bias of the first/second moment
estimates via the (1 - beta^t) terms. This variant REMOVES that correction,
isolating its effect (which matters most early in training).

Update rule (per parameter, step t, lr alpha, betas b1/b2, eps):
    m_t = b1 * m_{t-1} + (1 - b1) * g_t
    v_t = b2 * v_{t-1} + (1 - b2) * g_t^2
    theta_t = theta_{t-1} - alpha * m_t / (sqrt(v_t) + eps)     # NO /(1-b^t)

Reference for the standard rule this modifies:
    Kingma & Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015.
Document this file in the REPRO sheet as the custom-optimizer disclosure.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class AdamNoBiasCorrection(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if lr <= 0:
            raise ValueError(f"invalid lr {lr}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr)   # no bias correction
        return loss
