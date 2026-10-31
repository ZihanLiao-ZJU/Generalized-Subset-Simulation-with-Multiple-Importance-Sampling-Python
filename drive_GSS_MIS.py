"""Run the two GSS-MIS benchmarks using the shared sequential framework."""
import argparse
import json
import numpy as np
from GSS_MIS import GSS_MIS
from SeMCS import SeMCS
from Support_Fun.CW_aMH import CW_aMH
from Support_Fun.MIS import MIS
from Limit_state_functions.Ex1 import Ex1
from Limit_state_functions.Ex2 import Ex2
from Support_Fun.random import Random
from Support_Fun.results import summarize


def drive_GSS_MIS(example=1, N=None, repeats=100, seed=2026):
    if isinstance(example, str) and example.startswith("Ex"):
        example = int(example[2:])
    if example not in (1, 2) or repeats < 1:
        raise ValueError("Choose example 1 or 2 and positive repeats")
    N = (440, 410)[example - 1] if N is None else N
    if N < 20:
        raise ValueError("N must be at least 20")
    records = []
    for repetition in range(repeats):
        pf = (Ex1 if example == 1 else Ex2)()
        model = GSS_MIS(pf)
        sampler = CW_aMH(model, Random(seed + repetition))
        engine = SeMCS(sampler)
        engine.NumSam = N
        engine.MaxIte = 1000
        out = engine.RunIte()
        probability = MIS(out).Pf_hat
        assert pf.joint_count == int(sum(out["Ncal"]))
        records.append(dict(seed=seed + repetition, probability=probability.tolist(),
                            joint_calls=int(sum(out["Ncal"])), layer_joint_calls=[out["Ncal"].tolist()],
                            actual_level_samples=[out["Nsam"].tolist()], completed=out["terminal"]))
    return summarize(example, "GSS-MIS", N, repeats, seed, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=int, choices=[1, 2], default=1)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = drive_GSS_MIS(args.example, args.samples, args.repeats, args.seed)
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
