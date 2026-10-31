"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Arrays store coordinates/functions by row and samples by column.
"""

import numpy as np

EPS = np.finfo(float).eps


def mround(x):
    return np.floor(np.asarray(x) + 0.5).astype(int)


def mstd(x):
    return np.std(x, axis=1, ddof=1) if x.shape[1] > 1 else np.zeros(x.shape[0])


def quantile(x, p):
    return np.quantile(x, p, axis=1, method="hazen")


def logprior(x):
    with np.errstate(divide="ignore", under="ignore"):
        return np.log(np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi))


def source_logsum(a, axis=None):
    """Original logsum real-input behavior, including mixed NaNs and Inf overrides."""
    a = np.asarray(a)
    m = np.max(np.where(np.isnan(a), -np.inf, a), axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    z = np.where(np.isneginf(m), -np.inf, z)
    z = np.where(np.isposinf(m), np.inf, z)
    return np.squeeze(z) if axis is None else np.squeeze(z, axis=axis)
