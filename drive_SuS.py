"""Drive scalar SuS benchmarks through SeMCS, using independent component chains."""
import argparse
import json
import numpy as np
from SuS import SuS
from SeMCS import SeMCS
from Support_Fun.ScalarExample import ScalarExample
from MCMC_samplers.CW_aMH import CW_aMH
from MCMC_samplers.aCS import aCS
from MCMC_samplers.HMC import HMC
from MCMC_samplers.ES import ES
from Support_Fun.random import Random
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


def drive_SuS(example=1, N=None, repeats=100, seed=2026, sampler_name="CW_aMH"):
    if isinstance(example, str) and example.startswith("Ex"):
        example = int(example[2:])
    if example not in (1, 2) or repeats < 1:
        raise ValueError("Choose example 1 or 2 and positive repeats")
    N = (240, 180)[example - 1] if N is None else N
    if N < 20:
        raise ValueError("N must be at least 20")
    records = []
    for repetition in range(repeats):
        probabilities, calls, terminal, sizes = [], [], [], []
        for component in range(2 * example):
            pf = ScalarExample(example, component)
            model = SuS(pf, "physical", "midpoint", False)
            sampler = {"CW_aMH": CW_aMH, "aCS": aCS, "HMC": HMC, "ES": ES}[sampler_name](model, Random(seed + repetition + 10000 * (component + 1)))
            engine = SeMCS(sampler)
            engine.NumSam = N
            engine.MaxIte = 1000
            out = engine.RunIte()
            probabilities.append(float(estimate_pf(out, model.clip_threshold)))
            assert pf.joint_count == int(sum(out["Ncal"]))
            calls.append(out["Ncal"].tolist())
            terminal.append(out["terminal"])
            sizes.append(out["Nsam"].tolist())
        records.append(dict(seed=seed + repetition, probability=probabilities,
                            joint_calls=sum(map(sum,calls)), layer_joint_calls=calls,
                            actual_level_samples=sizes, completed=all(terminal)))
    return summarize(example, "SuS", N, repeats, seed, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=int, choices=[1, 2], default=1)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output")
    parser.add_argument("--sampler", choices=["CW_aMH", "aCS", "HMC", "ES"], default="CW_aMH")
    args = parser.parse_args()
    result = drive_SuS(args.example, args.samples, args.repeats, args.seed, args.sampler)
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
