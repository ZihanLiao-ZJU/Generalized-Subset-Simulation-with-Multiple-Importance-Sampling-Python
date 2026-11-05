"""GSS-MIS helper corresponding to the MATLAB file of the same name.

Xia and Liao (2025), Computers & Structures 313, 107741.
https://doi.org/10.1016/j.compstruc.2025.107741
"""

import numpy as np
from Operations_on_Logarithmic_values.logsum import logsum


def WgtMis_Blc(y, G, H, c, nsam, level):
    """Balance-heuristic log weights for one sampled level across all proposals."""
    mask = np.any((y[None, 3:, :] < G[:, :, None]) & H[:, :, None], axis=1)
    cross = np.where(mask, 0.0, -np.inf) + y[2][None, :] - c[:, None]
    return (
        np.log(nsam[level])
        + cross[level]
        - logsum(cross + np.log(nsam)[:, None], axis=0)
    )
