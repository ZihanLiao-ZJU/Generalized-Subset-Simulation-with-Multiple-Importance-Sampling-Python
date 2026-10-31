"""GSS-MIS helper corresponding to the MATLAB file of the same name.

Xia and Liao (2025), Computers & Structures 313, 107741.
https://doi.org/10.1016/j.compstruc.2025.107741
"""

from .benchmark import Benchmark


class Ex1(Benchmark):
    """Two circular failure regions in two standard-normal coordinates."""

    def __init__(self, pf=None):
        super().__init__(1, pf=pf)

    def _formulas(self, x):
        a, b = x
        return (
            lambda: a * a + b * b - 0.01,
            lambda: (a - 0.4) ** 2 + (b - 0.4) ** 2 - 0.01,
        )
