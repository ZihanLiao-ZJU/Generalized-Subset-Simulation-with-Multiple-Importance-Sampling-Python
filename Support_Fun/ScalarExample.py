"""GSS-MIS helper corresponding to the MATLAB file of the same name.

Xia and Liao (2025), Computers & Structures 313, 107741.
https://doi.org/10.1016/j.compstruc.2025.107741
"""

import numpy as np
from Limit_state_functions.Ex1 import Ex1
from Limit_state_functions.Ex2 import Ex2


class ScalarExample:
    """Evaluate only one selected performance function; Python indices start at zero."""

    def __init__(self, example, pf):
        self.example = example
        self.pf = pf
        self.nfun = 1
        self.count = 0
        self.joint_count = 0
        self._model = (Ex1 if example == 1 else Ex2)()

    def evaluate(self, x):
        self.joint_count += x.shape[1]
        self.count += x.shape[1]
        return np.vstack([self._model._formulas(x)[self.pf]()])
