"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Balance heuristic: Veach and Guibas (1995), SIGGRAPH, 419-428,
doi:10.1145/218380.218498. Actual sample counts are used in the mixture.

Arrays store coordinates/functions by row and samples by column.
"""

import numpy as np
from Support_Fun.Operations_on_Logarithmic_values.logsum import logsum
from Support_Fun.WgtMis_Blc import WgtMis_Blc
from Support_Fun.MisInt import MisInt
from GSS_MIS import likelihood


class MIS:
    """Balance-heuristic reliability estimate from sampled intermediate levels."""

    def __init__(self, run):
        count = len(run["y"])
        c = np.zeros(count)
        ys = [y.copy() for y in run["y"]]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            for i in range(count - 1):
                weights = (
                    likelihood(ys[i][3:], run["g"][i + 1], run["h"][i + 1])
                    + ys[i][2]
                    - ys[i][0]
                    - ys[i][1]
                )
                weights -= np.nanmax(weights)
                weights[np.isnan(weights)] = -np.inf
                c[i + 1] = c[i] + logsum(weights) - np.log(len(weights))
                ys[i + 1][0] -= c[i + 1]
            nf = ys[0].shape[0] - 3
            perlevel = np.zeros((nf, count))
            wm_all = []
            G = np.vstack([np.broadcast_to(g, (nf,)) for g in run["g"][:count]])
            H = np.vstack([np.broadcast_to(h, (nf,)) for h in run["h"][:count]]).astype(bool)
            for i, y in enumerate(ys):
                wm = WgtMis_Blc(y, G, H, c, run["Nsam"], i)
                wm_all.append(wm)
                for j in range(nf):
                    wi = y[2] - y[0] - y[1]
                    wi = wi.copy()
                    wi[y[3 + j] >= 0] = -np.inf
                    wi[np.isnan(wi)] = -np.inf
                    perlevel[j, i] = MisInt(wm, wi)
            pf = np.exp(logsum(perlevel, axis=1))
        self.Pf_hat = pf
        self.log_z = c
        self.weights = wm_all
        self.Y = ys
