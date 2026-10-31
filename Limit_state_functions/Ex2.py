"""GSS-MIS helper corresponding to the MATLAB file of the same name.

Xia and Liao (2025), Computers & Structures 313, 107741.
https://doi.org/10.1016/j.compstruc.2025.107741
"""

import numpy as np
from .benchmark import Benchmark


class Ex2(Benchmark):
    """Two quadratic and two linear performance-function boundaries."""

    def __init__(self, pf=None):
        super().__init__(2, pf=pf)

    def _formulas(self, x):
        a, b = x
        return (
            lambda: 3 + (a - b) ** 2 / 10 - (a + b) / np.sqrt(2),
            lambda: a - b + 7 / np.sqrt(2),
            lambda: 3 + (a - b) ** 2 / 10 + (a + b) / np.sqrt(2),
            lambda: b - a + 7 / np.sqrt(2),
        )
