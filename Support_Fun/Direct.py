"""GSS-MIS helper corresponding to the MATLAB file of the same name.

Xia and Liao (2025), Computers & Structures 313, 107741.
https://doi.org/10.1016/j.compstruc.2025.107741
"""

import numpy as np


class Direct:
    """Independent standard-normal inputs for the supplied two-dimensional examples."""

    @staticmethod
    def logpdf(x):
        # Keep the original density-then-log evaluation and nonfinite behavior.
        with np.errstate(divide="ignore", under="ignore"):
            return np.log(np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi))

    @staticmethod
    def sample(rng, n):
        # Separate draws preserve the original random-number ordering.
        return np.vstack([rng.randn(1, n), rng.randn(1, n)])
