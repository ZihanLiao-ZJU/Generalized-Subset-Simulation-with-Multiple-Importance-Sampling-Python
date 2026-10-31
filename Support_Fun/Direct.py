"""Independent physical marginals and their Gaussian/Nataf coordinate map."""
import numpy as np
from scipy.special import ndtr

class Direct:
    def __init__(self, marginals):
        self.Marg_X = tuple(marginals)
        self.Ndim = len(self.Marg_X)
        if not self.Ndim:
            raise ValueError("At least one marginal is required")

    def U2X(self, u):
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[0] != self.Ndim:
            raise ValueError("Input must have shape (Ndim, Nsamples)")
        x = np.empty_like(u)
        for j, marginal in enumerate(self.Marg_X):
            # scipy frozen distributions; exact affine Normal map avoids tail loss.
            if getattr(getattr(marginal, "dist", None), "name", None) == "norm":
                x[j] = marginal.mean() + marginal.std()*u[j]
            elif getattr(getattr(marginal, "dist", None), "name", None) == "lognorm":
                args, kw = marginal.args, marginal.kwds
                shape = kw.get("s", args[0] if args else None)
                loc = kw.get("loc", args[1] if len(args)>1 else 0.)
                scale = kw.get("scale", args[2] if len(args)>2 else 1.)
                x[j] = loc + scale*np.exp(shape*u[j])
            else:
                x[j] = marginal.ppf(ndtr(u[j]))
        return x

    def EvlPDF(self, x, idim=None):
        x = np.asarray(x)
        if idim is not None:
            return self.Marg_X[idim].logpdf(x)
        return np.sum([m.logpdf(x[j]) for j, m in enumerate(self.Marg_X)], axis=0)

    def SamGen(self, n, rng=None):
        return np.vstack([m.rvs(size=n, random_state=rng) for m in self.Marg_X])

    @staticmethod
    def logpdf(x):
        # Standard-normal benchmark compatibility, including underflow policy.
        with np.errstate(divide="ignore", under="ignore"):
            return np.log(np.exp(-0.5*x*x)/np.sqrt(2*np.pi))

    @staticmethod
    def sample(rng, n, ndim):
        return np.vstack([rng.randn(1, n) for _ in range(ndim)])
