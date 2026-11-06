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


from Support_Fun.SedSlt import SedSlt


class GSS_MIS:
    """Union intermediate distribution; iteration is owned by SeMCS."""
    def __init__(self, PF):
        self.TF = PF
        self.Nfun = getattr(PF, "Nfun", getattr(PF, "nfun", 1)) + 3
        self.Ndim = PF.Ndim
        self.Ncal = self.X = self.Y = None
        self.G = dict(parameters=np.array(np.inf), p=0.1, L=np.eye(self.Ndim))
        self.G["active"] = np.array(True)
        self.p = 0.1
        self.Nsam = 1000

    def EvlY(self, x):
        obj = copy.copy(self)
        obj.X = x
        physical = self.TF.Rdis.U2X(x) if hasattr(self.TF, "Rdis") else x
        values = self.TF.EvlLSF(physical) if hasattr(self.TF, "EvlLSF") else self.TF.evaluate(physical)
        obj.Y = np.vstack((-0.5 * (self.Ndim * np.log(2 * np.pi) + np.sum(x * x, axis=0)), values))
        obj.Ncal = x.shape[1]
        return obj

    def EvlLKF(self):
        return likelihood(self.Y[1:], self.G, self.G["active"])

    def EvlPDF(self):
        return self.Y[0]

    def UpdObj(self, x, y, p=None):
        if p is not None:
            self.p = p
        from Support_Fun.numerics import quantile
        self.G = dict(self.G, parameters=np.maximum(quantile(y[3:], self.p), 0), p=self.p)
        self.G["active"] = self.G["active"] & (self.G["parameters"] > 0)
        return y, self.G, self, 0

    @property
    def FlgCvg(self):
        return not np.any(self.G["active"])

    RunOptions = dict(
        PresetMode="replace", Layout="flat", SeedInputs="state_rng", SeedFunction=SedSlt,
    )

