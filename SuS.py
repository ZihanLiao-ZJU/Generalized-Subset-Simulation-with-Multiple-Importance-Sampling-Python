"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Subset simulation: Au and Beck (2001), Probabilistic Engineering
Mechanics 16, 263-277, doi:10.1016/S0266-8920(01)00019-4.

Arrays store coordinates/functions by row and samples by column.
"""

import numpy as np
from Support_Fun.numerics import mround
from Support_Fun.SuS_model import SuS_model
from Support_Fun.CW_aMH import sample_cw
from GSS_MIS_model import select_seeds, Run


def run_sus(tf, N, rng, p=0.1, max_levels=1000):
    """Original IndLSF_IdtDrt path, NOT GSS with one PF. Classical product output."""
    assert tf.nfun == 1
    g = np.array(np.inf)
    h = np.array(np.nan)
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
    terminal = False
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
        g = SuS_model.threshold(y, p)
        gs.append(g.copy())
        hs.append(h.copy())
        xseed, yseed, idx = select_seeds(x, y, g, h, N, rng)
        chosen.append(idx)
        nc = xseed.shape[1]
        steps = int(mround(N / nc)) if nc else 0
        terminal = SuS_model.complete(g)
        if nc <= 0 or level + 1 == max_levels:
            raw = False
            break
        if terminal:
            raw = True
            break
    assert tf.count == sum(calls)
    r = Run(xs, ys, gs, hs, ns, sizes, calls, chosen, scales, raw, terminal, tf.count)
    pf = 1.0
    for i in range(len(ys) - 1):
        pf *= np.mean(ys[i][3] < gs[i + 1])
    pf *= np.mean(ys[-1][3] < 0)
    return r, pf
