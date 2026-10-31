"""Run the two numerical examples from Xia and Liao (2025).

DOI: 10.1016/j.compstruc.2025.107741. See the repository README for usage.
"""

import argparse
import json
import numpy as np
from GSS_MIS import run_gss
from MIS_Rel import estimate_mis
from SuS import run_sus
from Limit_state_functions.benchmark import Benchmark
from Support_Fun.random import Random


def drive_GSS_MIS(example=1, N=None, repeats=100, seed=2026, algorithm="GSS-MIS"):
    """Return all replicates and summaries without excluding unfinished runs.

    One evaluation of all performance functions at one point counts once.
    SuS uses independent single-function chains and sums their evaluation counts.
    The empirical CoV uses the sample standard deviation across repetitions.
    """
    if example not in (1, 2) or repeats < 1 or algorithm not in ("GSS-MIS", "SuS"):
        raise ValueError("Choose example 1 or 2, positive repeats, and GSS-MIS or SuS")
    if N is None:
        N = ((440, 410) if algorithm == "GSS-MIS" else (240, 180))[example - 1]
    if N < 20:
        raise ValueError("N must be at least 20")
    records = []
    for repetition in range(repeats):
        current_seed = int(seed + repetition)
        components = [None] if algorithm == "GSS-MIS" else range(2 * example)
        probabilities, calls, terminal, actual_sizes = [], [], [], []
        for component in components:
            model = Benchmark(example, pf=component)
            rng = Random(
                current_seed
                if component is None
                else current_seed + 10000 * (component + 1)
            )
            if algorithm == "GSS-MIS":
                run = run_gss(model, N, rng)
                pf, _ = estimate_mis(run)
                probabilities.extend(pf.tolist())
            else:
                run, pf = run_sus(model, N, rng)
                probabilities.append(float(pf))
            assert model.joint_count == sum(run.ncal)
            assert model.count == model.nfun * model.joint_count
            calls.append(run.ncal)
            terminal.append(bool(run.terminal))
            actual_sizes.append(run.nsam)
        records.append(
            dict(
                seed=current_seed,
                probability=probabilities,
                joint_calls=sum(map(sum, calls)),
                layer_joint_calls=calls,
                actual_level_samples=actual_sizes,
                completed=all(terminal),
            )
        )
    estimates = np.asarray([row["probability"] for row in records])
    mean = estimates.mean(axis=0)
    std = estimates.std(axis=0, ddof=1) if repeats > 1 else np.full(2 * example, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = std / mean
    return dict(
        example=example,
        algorithm=algorithm,
        N=N,
        repeats=repeats,
        mean=mean.tolist(),
        standard_deviation=std.tolist(),
        cov=cov.tolist(),
        mean_joint_calls=float(np.mean([r["joint_calls"] for r in records])),
        unfinished=sum(not r["completed"] for r in records),
        records=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=int, choices=[1, 2], default=1)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--algorithm", choices=["GSS-MIS", "SuS"], default="GSS-MIS")
    parser.add_argument(
        "--output", help="Optional JSON file for complete replicate results"
    )
    args = parser.parse_args()
    result = drive_GSS_MIS(
        args.example, args.samples, args.repeats, args.seed, args.algorithm
    )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
