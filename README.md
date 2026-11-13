# GSuS-MIS: Generalized Subset Simulation with Multiple Importance Sampling (Python)

Python implementation of **generalized subset simulation with multiple importance sampling (GSuS-MIS)** for estimating small failure probabilities of multiple stochastic responses. This code accompanies the paper:

> Xia, W., & Liao, Z. (2025). Enhanced generalized subset simulation with multiple importance sampling for reliability estimation. *Computers & Structures, 313*, Article 107741. [https://doi.org/10.1016/j.compstruc.2025.107741](https://doi.org/10.1016/j.compstruc.2025.107741)

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

```text
.
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

## Quick start

Choose `Problem = Ex1` or `Problem = Ex2` in either driver. Both scripts define the sample count `N`, repeat count `Nrun` and `SamplerClass`. Available samplers are `CW_aMH` (default), `aCS`, `HMC` and `ES`.

Both scripts default to Example 1 and 100 runs. `drive_GSS_MIS` uses 440 samples per level and `drive_SuS` uses 240. For Example 2, set the sample counts to 410 and 180, respectively. For a single run, set `Nrun = 1` inside the script.

Run the chosen script from the repository root:

```bash
python drive_GSS_MIS.py
python drive_SuS.py
```

Use `drive_GSS_MIS` for GSuS-MIS or `drive_SuS` for the scalar SuS comparison.

Use multiple runs to estimate statistics across independent runs.

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
3. Construct `state = GSS_MIS(model)`, `sampler = CW_aMH(state)`,
   and `engine = SeMCS(sampler)`. Set `engine.NumSam = N` and call
   `result = engine.RunIte()`, followed by `MIS(result).Pf_hat`.

```python
from GSS_MIS import GSS_MIS
from SeMCS import SeMCS
from MCMC_samplers.CW_aMH import CW_aMH
from Support_Fun.MIS import MIS
from Limit_state_functions.Ex1 import Ex1

engine = SeMCS(CW_aMH(GSS_MIS(Ex1())))
engine.NumSam = 440
result = engine.RunIte()
print(MIS(result).Pf_hat)
```

`Direct(marginals)` accepts a list of SciPy frozen distributions for independent inputs. The performance function receives physical coordinates. Returned samples are stored in Gaussian coordinates; use `model.Rdis.U2X(u)` to obtain physical samples. The built-in standard-normal examples use the identity mapping.

## Outputs

The raw sampling history `g` contains `Nite` sampled states followed by one final candidate state. `Nsze` records the chain and step counts at each sampled level.

| Output | Meaning |
|---|---|
| `mean`, `standard_deviation` | Per-function mean and sample standard deviation across repetitions |
| `cov` | Empirical standard deviation divided by the mean |
| `mean_joint_calls` | Mean number of joint model evaluations |
| `unfinished` | Number of repetitions that did not complete all target events |
| `records` | Per-repetition probabilities, completion flags and counts |
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
- The scripts run repetitions serially and retain the results in memory.

## Citation

If you use this toolbox in your research, please cite:

> Xia, W., & Liao, Z. (2025). Enhanced generalized subset simulation with multiple importance sampling for reliability estimation. *Computers & Structures, 313*, Article 107741. [https://doi.org/10.1016/j.compstruc.2025.107741](https://doi.org/10.1016/j.compstruc.2025.107741)

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

