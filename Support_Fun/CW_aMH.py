"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Component-wise Metropolis sampling: Au and Beck (2001), Probabilistic
Engineering Mechanics 16, 263-277, doi:10.1016/S0266-8920(01)00019-4.

Arrays store coordinates/functions by row and samples by column.
"""

import numpy as np
from Support_Fun.numerics import EPS, mstd
from Support_Fun.Direct import Direct
from .SedSlt import SedSlt


def sample_cw(tf, rng, *, N=None, xseed=None, yseed=None, steps=1, evaluator=None):
    evaluate = evaluator
    # Uniform component proposals use the seed sample standard deviation.
    # The scale remains fixed within each level.
    initial = xseed is None
    if initial:
        nc = int(N)
        xseed = Direct.sample(rng, nc)
        yseed = evaluate(xseed)
        calls = nc
    else:
        xseed = xseed.copy()
        yseed = yseed.copy()
        nc = xseed.shape[1]
        calls = 0
    x = np.zeros((2, nc, steps))
    y = np.zeros((tf.nfun + 3, nc, steps))
    du = 2 * rng.rand(2, nc, steps) - 1
    p = rng.rand(2, nc, steps)
    sigma = np.maximum(mstd(xseed), EPS)
    for k in range(steps):
        if k == 0 and initial:
            x[:, :, k] = xseed
            y[:, :, k] = yseed
            continue
        for d in range(2):
            candidate = xseed[d] + sigma[d] * du[d, :, k]
            with np.errstate(over="ignore", invalid="ignore"):
                reject = np.exp(Direct.logpdf(candidate) - Direct.logpdf(xseed[d])) < p[d, :, k]
            candidate[reject] = xseed[d, reject]
            x[d, :, k] = candidate
        unchanged = np.all(x[:, :, k] == xseed, axis=0)
        y[:, unchanged, k] = yseed[:, unchanged]
        ids = np.flatnonzero(~unchanged)
        n = len(ids)
        if n:
            yy = evaluate(x[:, ids, k])
            y[:, ids, k] = yy
            with np.errstate(invalid="ignore"):
                reject = np.exp(yy[0] - yseed[0, ids]) < rng.rand(1, n).ravel()
            rejected = ids[reject]
            x[:, rejected, k] = xseed[:, rejected]
            y[:, rejected, k] = yseed[:, rejected]
            calls += n
        xseed = x[:, :, k].copy()
        yseed = y[:, :, k].copy()
    return (
        x.reshape(2, -1, order="F"),
        y.reshape(tf.nfun + 3, -1, order="F"),
        calls,
        sigma,
    )


class CW_aMH:
    """Componentwise sampler carrying its intermediate distribution in BayStc."""
    def __init__(self, BayStc, rng):
        self.BayStc, self.rng = BayStc, rng

    def SamGen(self, *args):
        model = self.BayStc
        if len(args) == 2:
            kwargs = dict(N=args[0], steps=args[1])
        else:
            kwargs = dict(xseed=args[0], yseed=args[1], steps=args[2])
        def evaluate(x):
            evaluated = model.EvlY(x)
            return np.vstack((evaluated.EvlLKF(), evaluated.EvlPDF(), evaluated.Y))
        x, y, calls, self.sigma = sample_cw(model.TF, self.rng,
                                          evaluator=evaluate, **kwargs)
        return x, y, calls, model

    def SltSed(self, x, y, N):
        model = self.BayStc
        return SedSlt(x, y, model.G, model.H, N, self.rng)
