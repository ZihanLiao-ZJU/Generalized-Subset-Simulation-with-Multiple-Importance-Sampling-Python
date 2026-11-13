"""Edit the settings below, then run this script from the repository root."""

import json
from GSS_MIS import GSS_MIS
from SeMCS import SeMCS
from MCMC_samplers.CW_aMH import CW_aMH
from MCMC_samplers.aCS import aCS
from MCMC_samplers.HMC import HMC
from MCMC_samplers.ES import ES
from Support_Fun.MIS import MIS
from Limit_state_functions.Ex1 import Ex1
from Limit_state_functions.Ex2 import Ex2
from Support_Fun.results import summarize


# %% Problem and sampling settings
Problem = Ex1               # Ex1 or Ex2
N = 440                     # Constant samples per level
Nrun = 100                  # Set to 1 for a single run
SamplerClass = CW_aMH       # CW_aMH, aCS, HMC or ES

# %% Run simulations
if N < 20 or Nrun < 1:
    raise ValueError("Use at least 20 samples and one run.")
records = []
for repetition in range(Nrun):
    pf = Problem()
    model = GSS_MIS(pf)
    sampler = SamplerClass(model)
    engine = SeMCS(sampler)
    engine.NumSam = N
    out = engine.RunIte()
    probability = MIS(out).Pf_hat
    records.append(
        dict(
            probability=probability.tolist(),
            joint_calls=int(sum(out["Ncal"])),
            layer_joint_calls=[out["Ncal"].tolist()],
            actual_level_samples=[out["Nsam"].tolist()],
            completed=out["flg_cvg"],
        )
    )
results = summarize(getattr(pf, "example", type(pf).__name__), "GSS-MIS", N, Nrun, records)

print(json.dumps({k: v for k, v in results.items() if k != "records"}, indent=2))
