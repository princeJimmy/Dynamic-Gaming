# Dynamic Competitive Insurance Pricing Game

Numerical research code for a finite-horizon insurance duopoly with dynamic customer portfolios, CARA preferences, Gamma-distributed claims, and price-sensitive competition.

> 中文简介：本仓库提供动态保险双寡头定价模型的数值实现，包括均衡求解、开环与闭环结果核验，以及单方参数敏感性分析。两个脚本可直接复现实验和图形；模型与数值细节见 [docs/MODEL_AND_NUMERICS.md](docs/MODEL_AND_NUMERICS.md)。

## What this repository contains

| File | Purpose |
| --- | --- |
| `research_coding_eq.py` | Solves the closed-loop equilibrium by backward induction, independently solves the open-loop system, compares both solutions, checks first-order conditions and gradients, and plots three numerical examples. |
| `research_par_sen.py` | Varies player *i*'s risk aversion, price sensitivity, and expense rate while keeping player *j* at the baseline; reports directional effects and produces sensitivity figures. |
| `docs/MODEL_AND_NUMERICS.md` | Describes the model ingredients, numerical algorithms, validation checks, parameter ranges, and interpretation caveats. |
| `requirements.txt` | Lists the Python dependencies. |

## Main features

- Finite-horizon, two-player dynamic pricing game
- CARA preferences with Gamma claim severities
- Compact feasible premium set `P = [0.001, 500]`
- Closed-loop solution by backward induction
- Independent bounded open-loop solution
- Projected best-response residuals, multistart checks, finite-difference gradient verification, and FOC diagnostics
- Asymmetric comparative statics for `lambda_i`, `a_i`, and `e_i`
- Publication-ready PNG output at 150 dpi

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/princeJimmy/Dynamic-Gaming.git
cd Dynamic-Gaming
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the equilibrium examples

```bash
python research_coding_eq.py
```

The script runs three examples, prints numerical diagnostics, and saves:

- `three_examples_premiums.png`
- optional `detailed_*.png` figures when `PLOT_DETAILED = True`

The equilibrium program checks:

1. closed-loop first-order conditions;
2. analytical gradients against centered finite differences;
3. the open-loop solution from a neutral initial path;
4. stability across randomly seeded open-loop starts; and
5. the maximum difference between open-loop and closed-loop premium paths.

## Reproduce the sensitivity analysis

```bash
python research_par_sen.py
```

The script saves:

- `asym_sensitivity_lambda.png`
- `asym_sensitivity_a.png`
- `asym_sensitivity_e.png`

Only player *i*'s selected parameter changes in each experiment. Player *j* remains at the baseline, but its equilibrium premium is endogenous and may respond through competition.

## Baseline specification

| Quantity | Baseline |
| --- | ---: |
| Horizon `T` | 5 |
| Premium set | `[0.001, 500]` |
| Expense rates `e_i, e_j` | 0.07, 0.07 |
| Risk aversion `lambda_i, lambda_j` | 0.004, 0.004 |
| Price sensitivity `a_i, a_j` | 1.7834, 1.7834 |
| Claim model | `Gamma(3, 0.03)` for both players |

The code uses the Gamma rate parameterization, so the baseline mean claim size is `3 / 0.03 = 100`. Risk integrability requires `lambda_k < beta_k`.

## Interpreting comparative statics

Sensitivity conclusions concern the reported finite grid and the selected summary statistic. A rise in the mean premium does not imply that every period's premium rises. For example, at a high expense rate the first-period premium can fall because the rival's endogenous price response reduces the strategic-complementarity term, even while the five-period mean and oscillation amplitude increase.

Boundary-active results should also be interpreted separately: once a premium reaches `0.001` or `500`, the projected equilibrium reflects the imposed feasible set.

## Headless or automated runs

The scripts call `plt.show()`. On a server without a display, use a non-interactive backend:

```bash
MPLBACKEND=Agg python research_coding_eq.py
MPLBACKEND=Agg python research_par_sen.py
```

## Custom experiments

- Edit the `examples` list in `research_coding_eq.py` to change equilibrium examples.
- Edit `BASELINE` or the parameter grids in `research_par_sen.py` to change sensitivity experiments.
- Keep `0 <= e_k < 1`, `lambda_k > 0`, `a_k > 0`, and `lambda_k < beta_k`.
- Review boundary warnings whenever the parameter grid is expanded.

## Research status

This repository is a numerical research companion. Results depend on the stated model, parameterization, equilibrium selection, tolerances, and finite parameter grids. Review the console diagnostics before using generated values in a manuscript.

## Citation and license

If you use this code, please cite the associated manuscript and this repository. A formal citation file and an open-source license can be added once the manuscript metadata and intended license are finalized.
