"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Arrays store coordinates/functions by row and samples by column.
"""

import numpy as np
from Support_Fun.numerics import mround, quantile
from Support_Fun.CW_aMH import sample_cw
from GSS_MIS_model import select_seeds, Run


def run_gss(tf, N, rng, p=0.1, max_levels=1000):
    g = np.array(np.inf)
    h = np.array(True)
    xs = []
    ys = []
    gs = [g.copy()]
    hs = [h.copy()]
    ns = []
    sizes = []
    calls = []
    chosen = []
    scales = []
    xseed = None
    yseed = None
    nc = N
    steps = 1
    raw = False
    # Source checks ite+1==MaxIte after each update, ending at MaxIte-1.
    for level in range(1, max_levels):
        x, y, ncal, scale = sample_cw(
            tf,
            g,
            h,
            rng,
            N=N if level == 1 else None,
            xseed=xseed,
            yseed=yseed,
            steps=steps,
        )
        xs.append(x)
        ys.append(y)
        ns.append(nc * steps)
        sizes.append((nc, steps))
        calls.append(ncal)
        scales.append(scale)
        g = np.maximum(quantile(y[3:], p), 0)
        h = h & (g > 0)
        gs.append(g.copy())
        hs.append(h.copy())
        xseed, yseed, idx = select_seeds(x, y, g, h, N, rng)
        chosen.append(idx)
        nc = xseed.shape[1]
        steps = int(mround(N / nc)) if nc else 0
        if nc <= 0 or level + 1 == max_levels:
            raw = False
            break
        if not np.any(h):
            raw = True
            break
    assert tf.count == tf.nfun * sum(calls)
    return Run(
        xs, ys, gs, hs, ns, sizes, calls, chosen, scales, raw, not np.any(h), tf.count
    )
