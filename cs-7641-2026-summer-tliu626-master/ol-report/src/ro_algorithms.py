"""
Randomized optimization algorithms for Part 1: RHC, SA, GA.

All three optimize the SAME validation-loss objective over the SAME flat vector
of final-layer parameters, and are compared by FUNCTION EVALUATIONS (each
objective call == 1 function evaluation). GA is compared by function evaluations,
not generations (a generation of pop P costs ~P evaluations).

Each algorithm returns (best_vector, best_loss, history) where history is a list
of dicts: {"func_evals", "best_loss", "wall_clock"} for best-so-far curves.

Required disclosures live in the function signatures/defaults and must be echoed
into the REPRO sheet:
  RHC: restart policy, perturbation distribution/scale, step-size schedule, stop
  SA : initial temperature, cooling schedule, step-size cooling, accept rule, stop
  GA : population size, selection, crossover, mutation rate/dist, elitism, stop
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

Objective = Callable[[np.ndarray], float]   # vector -> validation loss (minimize)


class FuncEvalCounter:
    """Wraps an objective to count and time function evaluations."""

    def __init__(self, objective: Objective, budget: int):
        self._obj = objective
        self.budget = budget
        self.n = 0
        self.t0 = time.perf_counter()

    def __call__(self, x: np.ndarray) -> float:
        self.n += 1
        return self._obj(x)

    @property
    def exhausted(self) -> bool:
        return self.n >= self.budget

    def stamp(self, best_loss: float) -> dict:
        return {"func_evals": self.n, "best_loss": best_loss,
                "wall_clock": time.perf_counter() - self.t0}


# ── Randomized Hill Climbing ──────────────────────────────────────────────────
def rhc(
    objective: Objective, x0: np.ndarray, budget: int, *,
    step: float = 0.05, n_restarts: int = 3, seed: int = 0,
) -> tuple[np.ndarray, float, list[dict]]:
    """Greedy local search with Gaussian perturbations and random restarts.

    Disclosures: perturbation ~ N(0, step^2 I) (fixed step schedule); restart
    policy = evenly split budget across n_restarts+1 climbs; stop = budget spent.
    """
    rng = np.random.default_rng(seed)
    counter = FuncEvalCounter(objective, budget)
    dim = x0.size
    per = max(1, budget // (n_restarts + 1))

    global_best_x, global_best = x0.copy(), counter(x0)
    history = [counter.stamp(global_best)]

    for r in range(n_restarts + 1):
        cur = x0.copy() if r == 0 else global_best_x + rng.normal(0, step * 3, dim)
        cur_loss = counter(cur)
        for _ in range(per):
            if counter.exhausted:
                break
            cand = cur + rng.normal(0, step, dim)
            cand_loss = counter(cand)
            if cand_loss < cur_loss:
                cur, cur_loss = cand, cand_loss
                if cur_loss < global_best:
                    global_best_x, global_best = cur.copy(), cur_loss
            history.append(counter.stamp(global_best))
        if counter.exhausted:
            break
    return global_best_x, global_best, history


# ── Simulated Annealing ───────────────────────────────────────────────────────
def sa(
    objective: Objective, x0: np.ndarray, budget: int, *,
    step: float = 0.05, t_init: float = 1.0, cooling: float = 0.995,
    step_cooling: float = 1.0, seed: int = 0,
) -> tuple[np.ndarray, float, list[dict]]:
    """Local search accepting worse moves with prob exp(-delta/T); geometric cooling.

    Disclosures: T_0=t_init, T <- cooling*T each step, optional step<-step_cooling*step,
    Metropolis acceptance, stop = budget spent.
    """
    rng = np.random.default_rng(seed)
    counter = FuncEvalCounter(objective, budget)
    dim = x0.size

    cur, cur_loss = x0.copy(), counter(x0)
    best_x, best = cur.copy(), cur_loss
    T, s = t_init, step
    history = [counter.stamp(best)]

    while not counter.exhausted:
        cand = cur + rng.normal(0, s, dim)
        cand_loss = counter(cand)
        delta = cand_loss - cur_loss
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            cur, cur_loss = cand, cand_loss
            if cur_loss < best:
                best_x, best = cur.copy(), cur_loss
        T *= cooling
        s *= step_cooling
        history.append(counter.stamp(best))
    return best_x, best, history


# ── Genetic Algorithm ─────────────────────────────────────────────────────────
def ga(
    objective: Objective, x0: np.ndarray, budget: int, *,
    pop_size: int = 30, mutation_rate: float = 0.1, mutation_std: float = 0.05,
    tournament_k: int = 3, elite: int = 2, init_std: float = 0.1, seed: int = 0,
) -> tuple[np.ndarray, float, list[dict]]:
    """Population search: tournament selection, uniform crossover, Gaussian mutation, elitism.

    Disclosures: pop_size, tournament_k selection, uniform crossover, per-gene
    mutation w.p. mutation_rate ~ N(0, mutation_std^2), elitism keeps `elite`
    best, stop = function-eval budget spent (NOT generation count).
    """
    rng = np.random.default_rng(seed)
    counter = FuncEvalCounter(objective, budget)
    dim = x0.size

    pop = np.vstack([x0] + [x0 + rng.normal(0, init_std, dim) for _ in range(pop_size - 1)])
    fitness = np.array([counter(ind) for ind in pop])
    best_i = int(fitness.argmin())
    best_x, best = pop[best_i].copy(), float(fitness[best_i])
    history = [counter.stamp(best)]

    def tournament() -> np.ndarray:
        idx = rng.integers(0, pop_size, tournament_k)
        return pop[idx[fitness[idx].argmin()]]

    while not counter.exhausted:
        order = np.argsort(fitness)
        new_pop = [pop[i].copy() for i in order[:elite]]     # elitism
        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            mask = rng.random(dim) < 0.5                     # uniform crossover
            child = np.where(mask, p1, p2)
            mut = rng.random(dim) < mutation_rate
            child[mut] += rng.normal(0, mutation_std, mut.sum())
            new_pop.append(child)

        pop = np.vstack(new_pop[:pop_size])
        fitness = np.array([counter(ind) for ind in pop])
        gen_best = int(fitness.argmin())
        if fitness[gen_best] < best:
            best_x, best = pop[gen_best].copy(), float(fitness[gen_best])
        history.append(counter.stamp(best))
        if counter.exhausted:
            break
    return best_x, best, history


RO_ALGORITHMS = {"rhc": rhc, "sa": sa, "ga": ga}
