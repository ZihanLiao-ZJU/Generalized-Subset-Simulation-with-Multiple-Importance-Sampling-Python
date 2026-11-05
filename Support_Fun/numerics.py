"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Arrays store coordinates/functions by row and samples by column.
"""

import numpy as np

def mround(x):
    return np.floor(np.asarray(x) + 0.5).astype(int)


def mstd(x):
    return np.std(x, axis=1, ddof=1) if x.shape[1] > 1 else np.zeros(x.shape[0])


def quantile(x, p):
    return np.quantile(x, p, axis=1, method="hazen")

