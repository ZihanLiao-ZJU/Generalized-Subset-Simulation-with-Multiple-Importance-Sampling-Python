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
from Support_Fun.random import Random
from Support_Fun.results import summarize


# %% Problem and sampling settings
example = 1
N = 440                     # Constant samples per level
Nrun = 100                  # Set to 1 for a single run
SamplerClass = CW_aMH       # CW_aMH, aCS, HMC or ES

# %% Run simulations
if example not in (1, 2) or N < 20 or Nrun < 1:
    raise ValueError("Choose example 1 or 2, N >= 20 and Nrun >= 1")
records = []
for repetition in range(Nrun):
    pf = (Ex1 if example == 1 else Ex2)()
    model = GSS_MIS(pf)
    sampler = SamplerClass(model, Random(None))
    engine = SeMCS(sampler)
    engine.NumSam = N
    engine.MaxIte = 1000
    out = engine.RunIte()
    probability = MIS(out).Pf_hat
    assert pf.joint_count == int(sum(out["Ncal"]))
    records.append(
        dict(
            probability=probability.tolist(),
            joint_calls=int(sum(out["Ncal"])),
            layer_joint_calls=[out["Ncal"].tolist()],
            actual_level_samples=[out["Nsam"].tolist()],
            completed=out["flg_cvg"],
        )
    )
results = summarize(example, "GSS-MIS", N, Nrun, records)

print(json.dumps({k: v for k, v in results.items() if k != "records"}, indent=2))
