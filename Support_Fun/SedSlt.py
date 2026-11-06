"""Select eligible seeds with the original chain-size and random-order rules."""
import numpy as np
from .numerics import mround


def support(f, g, h):
    g = g["parameters"] if isinstance(g, dict) else g
    g = np.broadcast_to(np.asarray(g), (f.shape[0],))
    h = np.broadcast_to(np.asarray(h, dtype=bool), (f.shape[0],))
    return np.any((f < g[:, None])[h], axis=0)


def likelihood(f, g, h):
    return np.where(support(f, g, h), 0.0, -np.inf)


def update_y(y, g, h):
    yy = y.copy()
    yy[0] = likelihood(yy[3:], g, h)
    yy[1] = yy[2]
    return yy


def SedSlt(x, y, g, N, rng):
    h = g["active"]
    ns = mround(N / np.arange(1, N + 1))
    nseed = mround(N / ns)
    scalar = np.all(np.isnan(h))
    eligible = np.flatnonzero(y[3] < g["parameters"] if scalar else support(y[3:], g, h))
    feasible = nseed[nseed <= len(eligible)]
    if not len(feasible):
        return x[:, :0], y[:, :0], np.array([], dtype=int)
    k = int(feasible[-1])
    selection = rng.randperm(len(eligible), k) if hasattr(rng, "randperm") else rng.permutation(len(eligible))[:k]
    chosen = eligible[selection]
    yy = y[:, chosen].copy()
    if scalar:
        yy[0] = np.where(yy[3] < g["parameters"], 0., -np.inf)
        yy[1] = yy[2]
    else:
        yy = update_y(yy, g, h)
    return x[:, chosen], yy, chosen
