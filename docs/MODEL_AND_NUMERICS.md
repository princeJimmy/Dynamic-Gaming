# Model and Numerical Notes

This document explains what the two research scripts compute, how the numerical checks are organized, and how to interpret the reported comparative statics. It is a code companion, not a substitute for the assumptions and proofs in the manuscript.

## 1. Model ingredients

There are two insurers, indexed by `i` and `j`, choosing premiums over a finite horizon. Each player has:

- CARA risk-aversion parameter `lambda_k > 0`;
- expense rate `0 <= e_k < 1`;
- price-sensitivity parameter `a_k > 0`;
- Gamma claim severity with shape `alpha_k` and rate `beta_k`; and
- a feasible premium set `[p_min, p_max]`.

The implementation uses the following quantities:

```text
log M_X(lambda) = -alpha * log(1 - lambda / beta)
kappa            = lambda * (1 - e)
log C(p)          = -kappa * p + log M_X(lambda)
```

The Gamma moment-generating function is finite only when `lambda < beta`. Both scripts validate this condition before solving the model.

The relative-price retention or growth multiplier is

```text
f_k(p_k, p_l) = exp[-a_k * (p_k - p_l) / p_l].
```

For a continuation scalar `Z_(k,t+1)`, the backward recursion is

```text
Z_(k,t) = f_k(p_(k,t), p_(l,t))
          * {exp[-kappa_k * p_(k,t)
                 + log M_Xk(lambda_k)
                 + Z_(k,t+1)] - 1}.
```

The interior analytical best response used by the stage solver is

```text
BR_k(p_l, Z_future)
  = {log M_Xk(lambda_k)
     + Z_future
     + log[1 + kappa_k * p_l / a_k]} / kappa_k.
```

The code projects this value onto `[p_min, p_max]`. Consequently, an equilibrium can be interior or boundary-active.

## 2. Closed-loop computation

`research_coding_eq.py` and `research_par_sen.py` solve the selected closed-loop equilibrium by backward induction:

1. Set the terminal continuation scalars to zero.
2. At period `t`, solve the two projected analytical best responses as a fixed point.
3. Store the equilibrium premiums.
4. Update each player's continuation scalar.
5. Repeat from the last period to the first.

The stage fixed-point iteration uses a tight tolerance and raises an error instead of silently returning a non-converged solution.

## 3. Independent open-loop computation

The equilibrium script also solves the full `2T`-dimensional open-loop system independently. It does not initialize the open-loop solver with the closed-loop path.

The implementation:

- starts from a neutral premium path unless the caller supplies both paths;
- constructs projected best-response residuals, which remain meaningful at a premium bound;
- uses SciPy's bounded trust-region least-squares algorithm;
- performs a short projected fixed-point polish for genuinely boundary-active coordinates;
- checks the optimizer's success flag and finiteness of the solution;
- requires the infinity norm of the projected residual to meet `1e-10`; and
- repeats the solve from five seeded random starting paths.

Agreement between the independently computed open-loop and closed-loop paths is a numerical consistency check for the examples. It supports the theoretical result but is not, by itself, a proof for every admissible parameter vector.

## 4. Numerical diagnostics

The equilibrium script reports several distinct checks.

### First-order conditions

For an interior coordinate, it compares

```text
C_k(p_k) * exp(Z_future)
```

with

```text
a_k / (a_k + kappa_k * p_l).
```

If a premium is at a bound, the projected residual is the appropriate diagnostic; an interior equality should not be imposed blindly.

### Gradient verification

The analytical gradient of the open-loop objective recursion is compared with adaptive centered finite differences at an off-equilibrium path. Checking away from the stationary point avoids relative-error measures being dominated by roundoff near zero.

### Multistart verification

Five reproducible random initial paths are compared with the neutral-start open-loop solution. This helps detect sensitivity to initialization or alternative numerical roots in the reported examples.

### Feasible-set checks

All best responses and open-loop variables use the same compact premium set:

```text
P = [0.001, 500].
```

When a parameter expansion activates either bound, the result should be labeled boundary-active and interpreted as a constrained equilibrium.

## 5. Equilibrium examples

The batch equilibrium program uses the following configurations.

| Example | T | Initial portfolios (i, j) | Expense rates (i, j) | Risk aversion (i, j) | Price sensitivity (i, j) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. Symmetric parameters | 5 | 2500, 1800 | 0.07, 0.07 | 0.004, 0.004 | 1.7834, 1.7834 |
| 2. Fully asymmetric | 6 | 2500, 1800 | 0.07, 0.12 | 0.003, 0.0009 | 1.7834, 2.0024 |
| 3. Asymmetric risk and expense | 5 | 2500, 2500 | 0.07, 0.12 | 0.003, 0.0009 | 1.7834, 1.7834 |

Unless changed in the code, both players use `Gamma(3, 0.03)` claims and `P = [0.001, 500]`.

## 6. Sensitivity design

`research_par_sen.py` changes one parameter of player `i` at a time. Every parameter of player `j` remains fixed at the baseline, although player `j`'s equilibrium premium still responds endogenously.

The implemented grids are:

| Parameter | Evaluated values |
| --- | --- |
| `lambda_i` | 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008 |
| `a_i` | 1.25, 1.50, 1.7834, 2.00, 2.25 |
| `e_i` | 0.00, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20 |

The script summarizes three objects:

- player `i`'s mean premium;
- player `j`'s mean premium as a competitive cross-effect; and
- player `i`'s oscillation amplitude, `max_t(p_i,t) - min_t(p_i,t)`.

A trend label therefore describes only these finite grids and summary statistics. It should not be read as a global derivative result.

## 7. Why the first-period price can fall at e_i = 0.20

Increasing `e_i` reduces `kappa_i = lambda_i(1 - e_i)`. Holding all other equilibrium objects fixed, this direct loading effect tends to raise player `i`'s premium.

However, the game is dynamic and prices are strategic complements. Changing `e_i` changes both continuation paths and the rival's equilibrium premiums. At the upper end of the reported expense grid, player `j` lowers its first-period premium sharply. This reduces player `i`'s competitive term

```text
log[1 + kappa_i * p_(j,1) / a_i].
```

That negative equilibrium-feedback effect can dominate the direct expense-loading effect in period 1. Thus the first-period premium may fall even while player `i`'s five-period mean premium and oscillation amplitude rise.

This is not caused by the premium projection in the baseline expense experiment: the reported `e_i = 0.20` path is not at either premium bound.

## 8. Extending the parameter range

When extending the experiments:

1. Keep `lambda_k < beta_k`; values close to `beta_k` make the Gamma MGF large and numerically demanding.
2. Preserve the common premium bounds in every solver.
3. Record whether any period is boundary-active.
4. Retain solver-success, projected-residual, FOC, gradient, and multistart checks.
5. Use a denser grid if a curve appears non-monotone or changes direction.
6. Report period-specific paths alongside means and amplitudes.
7. Treat a longer horizon as a new numerical experiment and re-check convergence.

## 9. Reproducibility checklist

Before copying a result into a paper or table, record:

- Python and dependency versions;
- the exact commit hash;
- all primitive parameters and the premium set;
- the horizon and parameter grid;
- whether any bound is active;
- the maximum projected residual;
- the open-loop multistart difference;
- the open-loop versus closed-loop difference; and
- the generated figure filename.

These records separate theoretical claims from evidence established only for a particular numerical experiment.
