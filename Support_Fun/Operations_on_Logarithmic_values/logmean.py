"""GSS-MIS helper corresponding to the MATLAB file of the same name.

Xia and Liao (2025), Computers & Structures 313, 107741.
https://doi.org/10.1016/j.compstruc.2025.107741
"""

import numpy as np
from .logsum import logsum


def logmean(a, axis=None):
    """Logarithm of the arithmetic mean, using the original logsum convention."""
    a = np.asarray(a)
    n = a.size if axis is None else a.shape[axis]
    return logsum(a, axis=axis) - np.log(n)
