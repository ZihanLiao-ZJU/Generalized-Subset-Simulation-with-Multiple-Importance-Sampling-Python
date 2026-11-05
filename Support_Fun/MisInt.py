"""GSS-MIS helper corresponding to the MATLAB file of the same name.

Xia and Liao (2025), Computers & Structures 313, 107741.
https://doi.org/10.1016/j.compstruc.2025.107741
"""

from Operations_on_Logarithmic_values.logmean import logmean


def MisInt(log_alpha, log_W):
    """One level's component-wise MIS integral in logarithmic form."""
    return logmean(log_alpha + log_W)
