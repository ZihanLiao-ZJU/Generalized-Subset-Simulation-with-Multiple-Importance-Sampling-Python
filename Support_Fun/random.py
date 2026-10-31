"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Arrays store coordinates/functions by row and samples by column.
"""

import numpy as np


class Random:
    """Independent Python RNG, with source call shapes; not MATLAB seeded parity."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def rand(self, *shape):
        return self.rng.random(np.prod(shape)).reshape(shape, order="F")

    def randn(self, *shape):
        return self.rng.standard_normal(np.prod(shape)).reshape(shape, order="F")

    def randperm(self, n, k):
        return self.rng.permutation(n)[:k]
