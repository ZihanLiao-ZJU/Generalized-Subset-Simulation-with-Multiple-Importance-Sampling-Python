"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Generalized subset simulation: Li, Ma and Cao (2015), Computers &
Structures 153, 239-251, doi:10.1016/j.compstruc.2014.10.014.

Arrays store coordinates/functions by row and samples by column.
"""

from dataclasses import dataclass
import numpy as np
from Support_Fun.numerics import mround, logprior


def support(f, g, h):
    g = np.broadcast_to(np.asarray(g), (f.shape[0],))
    h = np.broadcast_to(np.asarray(h, dtype=bool), (f.shape[0],))
    return np.any((f < g[:, None])[h], axis=0)


def likelihood(f, g, h):
    return np.where(support(f, g, h), 0.0, -np.inf)


def eval_y(tf, x, g, h):
    f = tf.evaluate(x)
    p = logprior(x).sum(axis=0)
    return np.vstack((likelihood(f, g, h), p, p, f))


def update_y(y, g, h):
    yy = y.copy()
    yy[0] = likelihood(yy[3:], g, h)
    yy[1] = yy[2]
    return yy


def select_seeds(x, y, g, h, N, rng):
    ns = mround(N / np.arange(1, N + 1))
    nseed = mround(N / ns)
    eligible = np.flatnonzero(support(y[3:], g, h))
    feasible = nseed[nseed <= len(eligible)]
    if not len(feasible):
        return x[:, :0], y[:, :0], np.array([], dtype=int)
    k = int(feasible[-1])
    chosen = eligible[rng.randperm(len(eligible), k)]
    return x[:, chosen], update_y(y[:, chosen], g, h), chosen


@dataclass
class Run:
    """Sampled levels, proposal supports, and independent completion diagnostics.

    ncal counts joint evaluations. scalar_calls records individual formulas.
    terminal reports that every target event has reached threshold zero.
    raw_flag retains the sampler loop status separately.
    """

    x: list
    y: list
    g: list
    h: list
    nsam: list
    nsze: list
    ncal: list
    selected: list
    sigma: list
    raw_flag: bool
    terminal: bool
    scalar_calls: int
