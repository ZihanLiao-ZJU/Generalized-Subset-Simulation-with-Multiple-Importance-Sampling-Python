"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Arrays store coordinates/functions by row and samples by column.
"""

from .benchmark import Benchmark


class Ex2(Benchmark):
    """Paper numerical example 2; optional zero-based single-PF selection."""

    def __init__(self, pf=None):
        super().__init__(2, pf=pf)
