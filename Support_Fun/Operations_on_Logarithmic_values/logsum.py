"""GSS-MIS helper corresponding to the MATLAB file of the same name.

Xia and Liao (2025), Computers & Structures 313, 107741.
https://doi.org/10.1016/j.compstruc.2025.107741
"""

import numpy as np


def logsum(a, axis=None):
    """Original logsum real-input behavior, including mixed NaNs and Inf overrides."""
    a = np.asarray(a)
    m = np.max(np.where(np.isnan(a), -np.inf, a), axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    z = np.where(np.isneginf(m), -np.inf, z)
    z = np.where(np.isposinf(m), np.inf, z)
    return np.squeeze(z) if axis is None else np.squeeze(z, axis=axis)
