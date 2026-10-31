"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Generalized subset simulation: Li, Ma and Cao (2015), Computers &
Structures 153, 239-251, doi:10.1016/j.compstruc.2014.10.014.

Arrays store coordinates/functions by row and samples by column.
"""

import copy
import numpy as np
from Support_Fun.Direct import Direct


def support(f, g, h):
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


class GSS_MIS:
    """Union intermediate distribution; iteration is owned by SeMCS."""
    def __init__(self, PF):
        self.TF = PF
        self.Nfun = PF.nfun + 3
        self.Ndim = 2
        self.Ncal = self.X = self.Y = None
        self.G = np.array(np.inf)
        self.H = np.array(True)
        self.p = 0.1
        self.Nsam = 1000

    def EvlY(self, x):
        obj = copy.copy(self)
        obj.X = x
        obj.Y = np.vstack((Direct.logpdf(x).sum(axis=0), self.TF.evaluate(x)))
        obj.Ncal = x.shape[1]
        return obj

    def EvlLKF(self):
        return likelihood(self.Y[1:], self.G, self.H)

    def EvlPDF(self):
        return self.Y[0]

    def UpdObj(self, x, y, p=None):
        if p is not None:
            self.p = p
        from Support_Fun.numerics import quantile
        self.G = np.maximum(quantile(y[3:], self.p), 0)
        self.H = self.H & (self.G > 0)
        return y, self.G, self, 0

    @property
    def FlgCvg(self):
        return not np.any(self.H)
