"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Arrays store coordinates/functions by row and samples by column.
"""

import numpy as np


class Benchmark:
    Ndim = 2

    def __init__(self, example, pf=None):
        self.example = example
        self.pf = pf
        self.nfun = 2 * example if pf is None else 1
        self.count = 0
        self.joint_count = 0

    def _formulas(self, x):
        # Backward-compatible Benchmark(example, pf=...) factory behavior.
        from .Ex1 import Ex1
        from .Ex2 import Ex2
        cls = Ex1 if self.example == 1 else Ex2
        return cls._formulas(self, x)

    def evaluate(self, x):
        self.joint_count += x.shape[1]
        formulas = self._formulas(x)
        ids = range(len(formulas)) if self.pf is None else [self.pf]
        self.count += len(ids) * x.shape[1]
        return np.vstack([formulas[i]() for i in ids])
