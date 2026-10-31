"""GSS-MIS helper corresponding to the MATLAB file of the same name.

Xia and Liao (2025), Computers & Structures 313, 107741.
https://doi.org/10.1016/j.compstruc.2025.107741
"""

import numpy as np
from .numerics import mround


class SuS_model:
    """Scalar threshold update and completion rule used by Subset Simulation."""

    @staticmethod
    def threshold(y, p):
        ordered = np.sort(y[3])
        k = int(mround(len(ordered) * p))
        if k < 1 or k >= len(ordered):
            raise IndexError("Source SuS midpoint index outside actual sample size")
        return np.array((ordered[k - 1] + ordered[k]) / 2)

    @staticmethod
    def complete(g):
        return bool(g <= 0)
