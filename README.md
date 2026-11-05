# GSuS-MIS: Generalized Subset Simulation with Multiple Importance Sampling (Python)

Python implementation of **generalized subset simulation with multiple importance sampling (GSuS-MIS)** for estimating small failure probabilities of multiple stochastic responses. This code accompanies the paper:

> **Enhanced generalized subset simulation with multiple importance sampling for reliability estimation.**
> Weili Xia and Zihan Liao. *Computers & Structures*, 313 (2025), 107741. doi: [10.1016/j.compstruc.2025.107741](https://doi.org/10.1016/j.compstruc.2025.107741)

Related implementation: [MATLAB](https://github.com/ZihanLiao-ZJU/Generalized-Subset-Simulation-with-Multiple-Importance-Sampling).

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Built-in examples](#built-in-examples)
- [Using a custom performance function](#using-a-custom-performance-function)
- [Outputs](#outputs)
- [Computational cost](#computational-cost)
- [Code-to-paper mapping](#code-to-paper-mapping)
- [Implementation notes](#implementation-notes)
- [Citation](#citation)
- [License](#license)

## Overview

GSuS-MIS estimates the failure probabilities

$$p_{f,j}=P\{g_j(\mathbf X)<0\}=\int f_{\mathbf X}(\mathbf x)\,I_{\{g_j(\mathbf x)<0\}}\,\mathrm d\mathbf x$$

by constructing a sequence of intermediate union events:

1. Performance-function quantiles define adaptive thresholds, and responses that reach the final failure threshold are removed from the active union.
2. Component-wise Metropolis MCMC generates samples in each intermediate event. Uniform proposal increments are scaled by the sample standard deviation of the current seeds.
3. Samples from all levels are combined with the balance-heuristic multiple importance sampling estimator. Historical thresholds, active responses, and actual level sample counts are retained in the weights.

Intermediate membership is strictly `g < threshold`, and final failure is strictly `g < 0`. The conditional quantile probability is `p = 0.1`; MIS calculations use logarithmic weights for numerical stability.

## Requirements

- Python 3.10 or later.
- NumPy 1.22 or later.

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Repository structure

Every toolbox includes the same `SeMCS(sampler).RunIte()` loop. Method classes own initialization, intermediate updates, seed selection and convergence. Fixed MCMC options are sampler attributes; evolving proposal scales and trajectory times are stored in the model's `G`. Each repository runs independently.

```text
├── GSS_MIS.py
├── LICENSE
├── Limit_state_functions/
│   ├── Ex1.py
│   ├── Ex2.py
│   ├── __init__.py
│   └── benchmark.py
├── MCMC_samplers/
│   ├── CW_aMH.py
│   ├── ES.py
│   ├── HMC.py
│   ├── MCMCBase.py
│   ├── __init__.py
│   └── aCS.py
├── Operations_on_Logarithmic_values/
│   ├── __init__.py
│   ├── logmean.py
│   └── logsum.py
├── README.md
├── SeMCS.py
├── SuS.py
├── Support_Fun/
│   ├── Direct.py
│   ├── MIS.py
│   ├── MisInt.py
│   ├── ScalarExample.py
│   ├── SedSlt.py
│   ├── WgtMis_Blc.py
│   ├── __init__.py
│   ├── numerics.py
│   ├── random.py
│   └── results.py
├── drive_GSS_MIS.py
├── drive_SuS.py
└── requirements.txt
```

`SeMCS` and `SuS` are shared with the SeMIS reliability Python toolbox.
The distribution classes define intermediate states and convergence, samplers
generate and select samples, `SeMCS.RunIte()` owns iteration, and the drivers
run repetitions and estimate probabilities. `Support_Fun/MIS.py` is a separate
postprocessor.

Both toolboxes use `SeMCS(sampler)` with the model stored in `sampler.BayStc`.
The drivers share `(example, N, repeats, seed)` and return summary dictionaries
with per-run `records`. Python's package and numerical helpers are retained only
where needed. Example and logarithmic helper implementations have matching MATLAB names.

## Quick start

Run the driver from the repository root:

```bash
python drive_GSS_MIS.py
```

By default this runs Example 1 with 100 independent repetitions and seed 2026. To select Example 2, save complete replicate results, or run SuS:

```bash
python drive_GSS_MIS.py --example 2 --repeats 100 --output results_ex2.json
python drive_SuS.py --example 2 --repeats 100
```

The driver can also be called from Python:

```python
from drive_GSS_MIS import drive_GSS_MIS
result = drive_GSS_MIS(example=1, N=440, repeats=100, seed=2026)
print(result["mean"], result["cov"], result["mean_joint_calls"])
```

The random generator is NumPy's `default_rng`. The same integer seed in Python and MATLAB does not imply identical random trajectories.

## Built-in examples

The two numerical examples correspond to the first two examples in the paper. Both use independent standard-normal inputs and samples arranged in a `2 × n` array.

| Example | Description | Dimension | Performance functions | GSuS-MIS samples/level | SuS samples/level |
|---|---|---|---|---|---|
| `Ex1` | Two circular failure regions, radius 0.1, centered at `(0, 0)` and `(0.4, 0.4)` | 2 | 2 | 440 | 240 |
| `Ex2` | Two quadratic and two linear boundaries, in the paper's equation order | 2 | 4 | 410 | 180 |

Only these two simple numerical examples are included. Integer chain lengths can change the realized number of samples in a level slightly; the MIS estimator uses the realized counts.

## Using a custom performance function

The samplers use Gaussian reference coordinates with dimension taken from the model. For a custom problem:

1. Follow `Limit_state_functions/Ex1.py` to provide `Ndim`, `nfun`, `count`, and an `evaluate(X)` method returning an array of shape `[nfun, nsamples]` for input `[Ndim, nsamples]`. Set `Rdis = Direct(marginals)` for independent physical marginals that are not standard normal.
2. Increment `count` by `nfun * nsamples` on every actual evaluation; optionally track `joint_count` by `nsamples` for reporting.
3. Construct `state = GSS_MIS(model)`, `sampler = CW_aMH(state, Random(seed))`,
   and `engine = SeMCS(sampler)`. Set `engine.NumSam = N` and call
   `result = engine.RunIte()`, followed by `MIS(result).Pf_hat`.

```python
from GSS_MIS import GSS_MIS
from SeMCS import SeMCS
from MCMC_samplers.CW_aMH import CW_aMH
from Support_Fun.MIS import MIS
from Support_Fun.random import Random
from Limit_state_functions.Ex1 import Ex1

engine = SeMCS(CW_aMH(GSS_MIS(Ex1()), Random(2026)))
engine.NumSam = 440
engine.MaxIte = 1000
result = engine.RunIte()
print(MIS(result).Pf_hat)
```

`NumSam` can also be a per-level vector; an exhausted finite schedule raises an
error rather than silently repeating its final size. Optional `G` and `H` lists
provide predefined intermediate states.

For nonnormal independent marginals, `Direct.U2X` maps Gaussian samples into the physical input model before evaluation; see the Gaussian/Nataf interface notes below.

## Outputs

| Output | Meaning |
|---|---|
| `mean`, `standard_deviation` | Per-function mean and sample standard deviation across repetitions |
| `cov` | Empirical standard deviation divided by the mean |
| `mean_joint_calls` | Mean number of joint model evaluations |
| `unfinished` | Number of repetitions that did not complete all target events |
| `records` | Per-repetition probabilities, seeds, completion flags and counts |
| `layer_joint_calls`, `actual_level_samples` within records | Actual evaluations and sample counts for each sampled level |

All repetitions are retained, including unfinished runs. Use at least two repetitions for empirical CoV; it is not a single-run error bound.

## Computational cost

**One evaluation of the complete performance-function vector at one input point counts as one joint model call.** Evaluated rejected candidates count; unchanged candidates whose values are reused do not. MIS post-processing uses already stored values.

The SuS baseline runs each performance function independently and sums the actual calls from those separate chains. Its total is not divided by the number of performance functions.

## Code-to-paper mapping

| Step | Code | Paper component |
|---|---|---|
| Intermediate union events | `GSS_MIS.UpdObj`, `SeMCS.RunIte` | Generalized subset-simulation construction |
| Component-wise Metropolis sampling | `MCMC_samplers.CW_aMH.CW_aMH.SamGen` | Conditional sampling within intermediate events |
| Recursive normalizing constants | `Support_Fun.MIS` | Intermediate importance-distribution normalization |
| Balance-heuristic MIS estimator | `Support_Fun.MIS`, `Support_Fun/WgtMis_Blc.py`, `Support_Fun/MisInt.py` | Combined failure-probability estimator, Eq. (18) |
| Numerical benchmarks | `Limit_state_functions` | First two numerical examples |


## Implementation notes

- Coordinate/function arrays use rows for dimensions or performance functions and columns for samples. MCMC arrays use `[Ndim, Nc, Ns]` and column-major flattening; Python indices start at zero.
- GSuS quantiles use NumPy's `hazen` method. Seed standard deviations use `ddof=1`, with zero spread for a single seed. Positive half-integer chain counts are rounded upward.
- The component-wise Metropolis scale is fixed within each level. The initial independent sample is retained; subsequent chain steps do not prepend the seeds.
- Historical proposal supports and realized sample counts are stored in the returned result dictionary and reused by the MIS estimator. Estimation does not mutate the stored function values.
- The driver accepts command-line options and runs repetitions serially. `--output` saves the complete replicate records as JSON.

The shared `SuS` class takes explicit coordinate, threshold, and clipping options.
`drive_SuS.py` uses `SuS(PF, "physical", "midpoint", False)` here; the SeMIS
reliability driver uses `SuS(PF, "standard_normal", "hazen", True)`. These retain
each benchmark's sampling convention without hidden repository-specific behavior.
The classical product estimator is computed in `drive_SuS`, not in the distribution class.

## Citation

If you use this code in your research, please cite:

> Xia, W., & Liao, Z. (2025). Enhanced generalized subset simulation with multiple importance sampling for reliability estimation. *Computers & Structures*, 313, 107741. https://doi.org/10.1016/j.compstruc.2025.107741

```bibtex
@article{xia2025enhanced,
  title   = {Enhanced generalized subset simulation with multiple importance sampling for reliability estimation},
  author  = {Xia, Weili and Liao, Zihan},
  journal = {Computers & Structures},
  volume  = {313},
  pages   = {107741},
  year    = {2025},
  doi     = {10.1016/j.compstruc.2025.107741}
}
```

Algorithm-specific references are included in the source: Li, Ma and Cao (2015) for generalized subset simulation and numerical benchmarks; Au and Beck (2001) for subset simulation; Veach and Guibas (1995) for the balance heuristic.

## License

Licensed under the [MIT License](LICENSE).


## Unified MCMC interface

Each standalone toolbox includes `CW_aMH`, `aCS`, `HMC`, and `ES`. The original
outer algorithm and default sampler remain unchanged; choose the sampler in the
driver (Python CLI: `--sampler`). Example: `drive_GSS_MIS(1, N=440, repeats=1, sampler_name="aCS")`.
All four expose `BayStc` and the same call:
`[u, y, Ncal, BayStc] = SamGen(u_seed, y_seed, Ns)` (Python returns a tuple).
`SamGen(Nchains, Nsteps)` initializes from the Gaussian reference. Arrays are
`[Ndim, Nchains, Nsteps]`; packed outputs are `[log_likelihood; log_reference; Y]`.
Dimension comes from the model, not from the two-dimensional examples.

Distribution state `G` carries model parameters and sampler adaptation. Legacy
numeric thresholds are stored in `G.parameters` (Python `G["parameters"]`);
existing named distribution parameters retain their names. aCS uses initial
`sigma0=1`, `lambda0=0.6`, five adaptation groups, and target acceptance 0.44.
The final `G.lambda` of one level initializes the next, rather than resetting.
ES uses the coordinatewise initial-angle `[0, pi]` version, with no stored seed
at the start of a conditional chain. HMC uses analytic Gaussian rotation in
Gaussian/Nataf reference coordinates, followed by the likelihood-factor MH test;
it is not a generic physical-coordinate gradient-HMC implementation. `G.L` or
`G.L_pi` is the Gaussian reference Cholesky factor, not the covariance matrix.
HMC's final trajectory time is likewise carried in `G.HMC_time`.

Custom nonnormal physical models must be mapped into Gaussian/Nataf coordinates
before these kernels are used. Arbitrary dimensional arrays are supported; the
supplied benchmark formulas retain their documented dimensions.


GSuS now samples and stores Gaussian coordinates `u` in every kernel. `Direct.U2X`
maps these into physical independent marginals before calling the performance
function; transform returned samples with that method when physical coordinates
are required. MATLAB accepts its original probability-distribution objects;
Python `Direct(marginals)` accepts a list of SciPy frozen distributions (for
example `scipy.stats.norm` and `scipy.stats.lognorm`) as `PF.Rdis`. Set `PF.Ndim`
and `PF.Nfun` (or Python `nfun`), and provide `EvlLSF` (or Python `evaluate`).
This is the independent-input Nataf case; no correlated-marginal extension is
implied. Standard-normal benchmark mappings remain the identity. The shared
SuS `physical` option selects this `PF.Rdis` mapping; it does not run MCMC in
untransformed physical space.
