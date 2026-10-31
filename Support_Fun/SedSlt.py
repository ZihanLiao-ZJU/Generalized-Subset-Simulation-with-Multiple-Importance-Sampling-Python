"""Select eligible seeds with the original chain-size and random-order rules."""
import numpy as np
from .numerics import mround
from GSS_MIS import support, update_y

def SedSlt(x, y, g, h, N, rng):
    ns = mround(N / np.arange(1, N + 1))
    nseed = mround(N / ns)
    eligible = np.flatnonzero(support(y[3:], g, h))
    feasible = nseed[nseed <= len(eligible)]
    if not len(feasible):
        return x[:, :0], y[:, :0], np.array([], dtype=int)
    k = int(feasible[-1])
    chosen = eligible[rng.randperm(len(eligible), k)]
    return x[:, chosen], update_y(y[:, chosen], g, h), chosen



