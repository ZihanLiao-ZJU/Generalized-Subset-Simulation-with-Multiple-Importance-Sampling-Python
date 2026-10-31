"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Arrays store coordinates/functions by row and samples by column.
"""

import numpy as np


class Benchmark:
    def __init__(self, example, pf=None):
        self.example = example
        self.pf = pf
        self.nfun = 2 * example if pf is None else 1
        self.count = 0
        self.joint_count = 0

    def evaluate(self, x):
        self.joint_count += x.shape[1]
        a, b = x
        if self.example == 1:
            formulas = (
                lambda: a * a + b * b - 0.01,
                lambda: (a - 0.4) ** 2 + (b - 0.4) ** 2 - 0.01,
            )
        else:
            formulas = (
                lambda: 3 + (a - b) ** 2 / 10 - (a + b) / np.sqrt(2),
                lambda: a - b + 7 / np.sqrt(2),
                lambda: 3 + (a - b) ** 2 / 10 + (a + b) / np.sqrt(2),
                lambda: b - a + 7 / np.sqrt(2),
            )
        ids = range(len(formulas)) if self.pf is None else [self.pf]
        self.count += len(ids) * x.shape[1]
        return np.vstack([formulas[i]() for i in ids])
