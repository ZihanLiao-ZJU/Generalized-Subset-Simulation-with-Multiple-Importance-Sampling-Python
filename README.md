# GSuS-MIS — Python

## Overview
Generalized subset simulation with multiple importance sampling estimates several failure probabilities using intermediate union events and a balance-heuristic MIS estimator. A separate SuS driver estimates each response independently with a probability product. The default sampler is `CW_aMH`. GSuS is the display name; existing `GSS_MIS` APIs and repository URLs are retained.

## Structure
- `drive_GSS_MIS.py`, `drive_SuS.py` — example selection, settings, repeated runs and summaries.
- `SeMCS.py` — sequential iteration; `GSS_MIS.py` / `SuS.py` — intermediate distributions, threshold updates and convergence.
- `Support_Fun/MIS.py` — postprocessing of all sampled GSuS levels.
- `Limit_state_functions/Ex1.py`, `Ex2.py` — bundled performance functions.
- `Support_Fun/` — locally included `CW_aMH`, `aCS`, `HMC`, `ES` and shared numerical helpers.

## Quick start
Run from this repository's root:

```bash
python -m pip install -r requirements.txt
python drive_GSS_MIS.py --example 1 --samples 440 --repeats 1 --seed 411 --sampler CW_aMH
python drive_SuS.py --example 2 --samples 180 --repeats 1 --seed 411 --sampler ES
```
The driver returns response-wise `mean`, `standard_deviation`, `cov`, actual `mean_joint_calls`, `unfinished` and individual `records`. A nonzero `unfinished` count must not be treated as completed validation. Both drivers support the same example/sampler selection.

## Examples and samplers
| Example | Model / dimension | Driver selector |
|---|---|---|
| 1 | Two circular failure regions; 2 independent standard-normal inputs, 2 outputs | `1` |
| 2 | Four performance functions (two quadratic, two linear); 2 independent standard-normal inputs, 4 outputs | `2` |

- Available samplers: `CW_aMH`, `aCS`, `HMC`, `ES`; default `CW_aMH`. The quick start shows explicit selection.
- Driver defaults remain example 1, 100 repetitions and seed 2026. GSuS uses N=440/410 for examples 1/2; SuS uses N=240/180. Intermediate probability is 0.1; the drivers set the level budget to 1000.
- Coordinates stored by the kernels are Gaussian reference coordinates. Models map them to physical inputs via `Rdis.U2X` (or the distribution selected by SuS). `Direct` supports independent marginals; it is not a general correlated-Nataf solver.
- Custom models provide their dimension and performance-function evaluations. Common samplers are not fixed to 2D; the supplied examples intentionally are 2D.

## Dependencies
- Python (tested with 3.12.14), NumPy ≥1.22 and SciPy ≥1.8; see `requirements.txt`.
- No cross-repository runtime import or external model data is required for the bundled examples.

## Method & validation notes
- **Implementation:** GSuS uses strict `g < threshold` union events, removes completed responses from the active union, and combines every sampled level with actual-count MIS weights. Final failure is `g < 0`. The scalar SuS driver preserves its midpoint-threshold/unclipped-final-threshold policy and actual conditional-probability product; these policies are not silently replaced by another toolbox's settings.
- **Sampler policy:** `CW_aMH` uses seed-spread-scaled **uniform random-walk increments in whitened Gaussian-reference coordinates**. `ES` is the coordinatewise initial-angle `[0, pi]` version, not a claim of identical historical scalar-angle ESS. `aCS` starts with `sigma0=1`, `lambda0=0.6` and carries the final lambda between levels (five adaptation groups, target acceptance 0.44). `HMC` uses analytic Gaussian-reference rotation; its adaptive trajectory time is carried between levels. It is not generic physical-coordinate gradient HMC.
- **Shared interface:** `SamGen(seed_u, seed_y, Nsteps)` returns `(u, y, Ncal, BayStc)` (MATLAB uses multiple outputs); arrays are dimension/function × chain × step. `G` carries thresholds and adaptation. Local copies of common kernels keep each repository independently runnable; no sibling repository is required.
- **Outputs:** SD is the sample SD across independent runs; CoV is SD/mean, not the standard error of the mean. One joint vector evaluation at one input counts as **one** model call; independent scalar SuS run costs are actually summed, not divided or multiplied by the number of responses.
- **Validation scope:** clean isolated example calls for both drivers and all four kernels, plus 3D/7D interface checks, passed. The latest cleanup leaves scientific source unchanged. Earlier original-source/framework and unified-MCMC revisions are distinct provenance stages, not a claim that only filenames differed from the earliest upload.
- **Limitations:** N=440/410 are reproduction/calibration settings, not explicitly reported original-paper GSuS sample counts. The paper's sampler identity is supported by supplied source rather than explicitly named in its text. Same-N runs need not have equal cost, and new-kernel/interface checks do not establish exact agreement with every paper statistic. No engineering examples beyond the two listed models are bundled.

## Reference & license
Weili Xia and Zihan Liao. **Enhanced generalized subset simulation with multiple importance sampling for reliability estimation.** *Computers & Structures* 313 (2025), 107741. [DOI](https://doi.org/10.1016/j.compstruc.2025.107741).

Generalized subset simulation: H.-S. Li, Y.-Z. Ma and Z. Cao (2015), *Computers & Structures* 153, 239–251. [DOI](https://doi.org/10.1016/j.compstruc.2014.10.014). Balance heuristic: Veach and Guibas (1995), [DOI](https://doi.org/10.1145/218380.218498).

Licensed under the [MIT License](LICENSE). Original copyright and source notices are retained. [Other language](https://github.com/ZihanLiao-ZJU/Generalized-Subset-Simulation-with-Multiple-Importance-Sampling).
