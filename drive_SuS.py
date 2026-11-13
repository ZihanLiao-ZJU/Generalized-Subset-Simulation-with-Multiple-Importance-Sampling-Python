"""Edit the settings below, then run this script from the repository root."""
import json
import numpy as np
from SuS import SuS
from SeMCS import SeMCS
from Limit_state_functions.Ex1 import Ex1
from Limit_state_functions.Ex2 import Ex2
from MCMC_samplers.CW_aMH import CW_aMH
from MCMC_samplers.aCS import aCS
from MCMC_samplers.HMC import HMC
from MCMC_samplers.ES import ES
from Support_Fun.results import summarize


def estimate_pf(result, clip_threshold=True):
    """Classical product estimator, using zero for an unclipped final target."""
    probability = 1.0
    for i, y in enumerate(result["y"]):
        values = np.asarray(y).reshape(np.shape(y)[0], -1, order="F")[3, :]
        threshold = (0 if i == len(result["y"]) - 1 and not clip_threshold
                     else result["g"][i + 1]["parameters"])
        probability *= np.mean(values < threshold)
    return probability


# %% Problem and sampling settings
Problem = Ex1               # Ex1 or Ex2
N = 240                     # Constant samples per level
Nrun = 100                  # Set to 1 for a single run
SamplerClass = CW_aMH       # CW_aMH, aCS, HMC or ES

# %% Run simulations
if N < 20 or Nrun < 1:
    raise ValueError("Use at least 20 samples and one run.")
example = Problem()
records = []
for repetition in range(Nrun):
    probabilities, calls, terminal, sizes = [], [], [], []
    for component in range(example.nfun):
        pf = Problem(pf=component)
        model = SuS(pf, "physical", "midpoint", False)
        sampler = SamplerClass(model)
        engine = SeMCS(sampler)
        engine.NumSam = N
        out = engine.RunIte()
        probabilities.append(float(estimate_pf(out, model.clip_threshold)))
        calls.append(out["Ncal"].tolist())
        terminal.append(out["flg_cvg"])
        sizes.append(out["Nsam"].tolist())
    records.append(
        dict(
            probability=probabilities,
            joint_calls=sum(map(sum, calls)),
            layer_joint_calls=calls,
            actual_level_samples=sizes,
            completed=all(terminal),
        )
    )
results = summarize(example.example, "SuS", N, Nrun, records)

print(json.dumps({k: v for k, v in results.items() if k != "records"}, indent=2))
