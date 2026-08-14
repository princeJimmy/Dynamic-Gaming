# -*- coding: utf-8 -*-
"""
Asymmetric Sensitivity Analysis for the Insurance Duopoly Model.
Varies player i's parameter only (lambda_i, a_i, e_i); player j stays at baseline.
This reproduces the experiments described in Section 5.3 and Table 2.
Uses the closed-loop backward-induction solver (equivalent to OL by Theorem 4.1).
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Common compact premium set used in the paper
DEFAULT_P_MIN = 1e-3
DEFAULT_P_MAX = 500.0

# Core Model Functions

def get_gamma_log_mgf(lam, alpha, beta):
    if lam <= 0 or alpha <= 0 or beta <= 0:
        raise ValueError(
            f"Need lambda, alpha, and beta > 0; got {lam}, {alpha}, {beta}."
        )
    if lam >= beta:
        raise ValueError(f"lambda={lam} >= beta={beta}")
    return -alpha * math.log1p(-lam / beta)

def f_func(p_own, p_opp, a):
    if p_own <= 0 or p_opp <= 0 or a <= 0:
        raise ValueError(
            f"Need positive premiums and a > 0; got {p_own}, {p_opp}, {a}."
        )
    return math.exp(-a * (p_own - p_opp) / p_opp)

def analytical_best_response(
    p_opp, Z_future, kappa, log_mgf, a,
    p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX,
):
    """Projected analytical best response -> Eq. (44)"""
    if p_opp <= 0 or kappa <= 0 or a <= 0:
        raise ValueError(
            f"Need p_opp, kappa, and a > 0; got {p_opp}, {kappa}, {a}."
        )
    if not (0 < p_min < p_max):
        raise ValueError(f"Need 0 < p_min < p_max; got [{p_min}, {p_max}].")
    val = (log_mgf + Z_future + math.log1p(kappa * p_opp / a)) / kappa
    return float(np.clip(val, p_min, p_max))

def solve_stage_nash(Zf_i, Zf_j, kappa_i, kappa_j, log_mgf_i, log_mgf_j, ai, aj,
                     p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX,
                     tol=1e-14, max_iter=10000):
    pi = float(np.clip(
        (log_mgf_i + Zf_i) / kappa_i * 1.3, p_min, p_max
    ))
    pj = float(np.clip(
        (log_mgf_j + Zf_j) / kappa_j * 1.3, p_min, p_max
    ))
    for _ in range(max_iter):
        pi_new = analytical_best_response(
            pj, Zf_i, kappa_i, log_mgf_i, ai, p_min, p_max
        )
        pj_new = analytical_best_response(
            pi_new, Zf_j, kappa_j, log_mgf_j, aj, p_min, p_max
        )
        if abs(pi_new - pi) < tol and abs(pj_new - pj) < tol:
            return pi_new, pj_new
        pi, pj = pi_new, pj_new
    raise RuntimeError(
        f"Stage Nash solver did not converge after {max_iter} iterations."
    )

def Z_step(p_own, p_opp, Z_future, a, kappa, log_mgf):
    f = f_func(p_own, p_opp, a)
    exponent = -kappa * p_own + log_mgf + Z_future
    if exponent > 700:
        raise FloatingPointError(
            f"Z recursion overflow: exp argument is {exponent:.6g}."
        )
    return f * math.expm1(exponent)

def solve_equilibrium(N, lam_i, lam_j, ai, aj, ei, ej,
                      alpha_i, beta_i, alpha_j, beta_j,
                      p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX):
    if not isinstance(N, int) or N < 1:
        raise ValueError(f"N must be a positive integer; got {N!r}.")
    if not (0 <= ei < 1 and 0 <= ej < 1):
        raise ValueError(f"Expense rates must lie in [0, 1); got {ei}, {ej}.")
    if ai <= 0 or aj <= 0:
        raise ValueError(f"Price sensitivities must be positive; got {ai}, {aj}.")
    if not (0 < p_min < p_max):
        raise ValueError(f"Need 0 < p_min < p_max; got [{p_min}, {p_max}].")

    kappa_i = lam_i * (1.0 - ei)
    kappa_j = lam_j * (1.0 - ej)
    log_mgf_i = get_gamma_log_mgf(lam_i, alpha_i, beta_i)
    log_mgf_j = get_gamma_log_mgf(lam_j, alpha_j, beta_j)

    prem_i = [0.0] * N
    prem_j = [0.0] * N
    Z_i = [0.0] * (N + 1)
    Z_j = [0.0] * (N + 1)

    for t in range(N, 0, -1):
        idx = t - 1
        pi_star, pj_star = solve_stage_nash(
            Z_i[t], Z_j[t],
            kappa_i, kappa_j, log_mgf_i, log_mgf_j, ai, aj,
            p_min, p_max
        )
        prem_i[idx] = pi_star
        prem_j[idx] = pj_star
        Z_i[idx] = Z_step(pi_star, pj_star, Z_i[t], ai, kappa_i, log_mgf_i)
        Z_j[idx] = Z_step(pj_star, pi_star, Z_j[t], aj, kappa_j, log_mgf_j)

    return prem_i, prem_j, Z_i, Z_j

# Baseline Parameters

BASELINE = {
    'N': 5,
    'lam_i': 0.004, 'lam_j': 0.004,
    'ai': 1.7834, 'aj': 1.7834,
    'ei': 0.07, 'ej': 0.07,
    'alpha_i': 3.0, 'beta_i': 0.03,
    'alpha_j': 3.0, 'beta_j': 0.03,
    'p_min': DEFAULT_P_MIN, 'p_max': DEFAULT_P_MAX,
}

def run_with(**overrides):
    p = {**BASELINE, **overrides}
    return solve_equilibrium(
        p['N'], p['lam_i'], p['lam_j'], p['ai'], p['aj'],
        p['ei'], p['ej'], p['alpha_i'], p['beta_i'], p['alpha_j'], p['beta_j'],
        p['p_min'], p['p_max']
    )


# Sensitivity routine (1x3 figure shown in Table 2)

def run_sensitivity(param_name, param_label, param_values, override_key,
                    cmap_name, filename):
    """
    Vary override_key for player i only; player j stays at baseline.
    Produces a 1x3 figure
    """
    N = BASELINE['N']
    periods = list(range(1, N + 1))

    results = []
    for val in param_values:
        try:
            pi, pj, Zi, Zj = run_with(**{override_key: val})
            boundary_i = any(
                np.isclose(p, BASELINE['p_min'], atol=1e-9)
                or np.isclose(p, BASELINE['p_max'], atol=1e-9)
                for p in pi
            )
            boundary_j = any(
                np.isclose(p, BASELINE['p_min'], atol=1e-9)
                or np.isclose(p, BASELINE['p_max'], atol=1e-9)
                for p in pj
            )
            results.append({
                'val': val, 'pi': pi, 'pj': pj,
                'boundary_i': boundary_i, 'boundary_j': boundary_j,
            })
        except (ValueError, OverflowError, FloatingPointError, RuntimeError) as exc:
            print(f"  Skipping {param_name}={val}: {exc}")

    if not results:
        print("  No valid results. Skipping plot.")
        return None

    boundary_values = [
        r['val'] for r in results if r['boundary_i'] or r['boundary_j']
    ]
    if boundary_values:
        print(
            f"  Boundary-active values for {param_name}: {boundary_values} "
            f"under P=[{BASELINE['p_min']}, {BASELINE['p_max']}]"
        )

    n_valid = len(results)
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(k / max(n_valid - 1, 1)) for k in range(n_valid)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        rf'Sensitivity to Player $i$ {param_label}'
        rf'  (Player $j$ at baseline)',
        fontsize=15, fontweight='bold'
    )

    # Player i's premium paths
    ax = axes[0]
    for r, c in zip(results, colors):
        ax.plot(periods, r['pi'], 'o-', color=c,
                label=rf'${param_name}={r["val"]}$', lw=2, ms=5)
    ax.set_xlabel('Period $t$', fontsize=11)
    ax.set_ylabel('$p_{i,t}^*$', fontsize=11)
    ax.set_title('Player $i$ Premium Path', fontweight='bold')
    ax.set_xticks(periods)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Mean premium: own effect AND competitive cross effect
    ax = axes[1]
    vals = [r['val'] for r in results]
    mean_i = [np.mean(r['pi']) for r in results]
    mean_j = [np.mean(r['pj']) for r in results]
    ax.plot(vals, mean_i, 'o-', color='tab:red', lw=2, ms=7,
            label=r'$\overline{p}_i^*$ (own)')
    ax.plot(vals, mean_j, 's-', color='tab:blue', lw=2, ms=7,
            label=r'$\overline{p}_j^*$ (competitor)')
    ax.set_xlabel(rf'${param_name}$', fontsize=11)
    ax.set_ylabel('Mean Premium', fontsize=11)
    ax.set_title('Mean Premium: Own vs Competitor', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Oscillation amplitude of player i
    ax = axes[2]
    amp_i = [max(r['pi']) - min(r['pi']) for r in results]
    ax.plot(vals, amp_i, 'D-', color='tab:green', lw=2, ms=7)
    ax.set_xlabel(rf'${param_name}$', fontsize=11)
    ax.set_ylabel(r'$\max_t p_{i,t}^* - \min_t p_{i,t}^*$', fontsize=11)
    ax.set_title('Player $i$ Oscillation Amplitude', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {filename}")

    # Directional summary for Table 2
    def trend(seq):
        diffs = np.diff(seq)
        if np.all(diffs > 1e-9):
            return "increasing (up)"
        if np.all(diffs < -1e-9):
            return "decreasing (down)"
        if np.allclose(diffs, 0, atol=1e-9):
            return "flat (no change)"
        return "non-monotone"

    summary = {
        'mean_i': trend(mean_i),
        'mean_j': trend(mean_j),
        'amp_i':  trend(amp_i),
    }
    print(f"\n  --- {param_label} directional summary (Table 2) ---")
    print(f"    Mean p_i^*           : {summary['mean_i']}")
    print(f"    Mean p_j^* (rival)   : {summary['mean_j']}")
    print(f"    Oscillation amp (i)  : {summary['amp_i']}")

    return {'results': results, 'summary': summary}

# Run All Three Sensitivity Analyses (ranges aligned with Section 5.3)

def main():
    print("=" * 65)
    print("  Asymmetric Sensitivity Analysis (Section 5.3 / Table 2)")
    print("  Vary Player i only; Player j fixed at baseline")
    print("=" * 65)
    print(f"\n  Baseline: N={BASELINE['N']}, "
          f"lambda={BASELINE['lam_i']}, a={BASELINE['ai']}, e={BASELINE['ei']}")
    print(f"  Premium set: [{BASELINE['p_min']}, {BASELINE['p_max']}]")
    print(f"  Claims: Gamma({BASELINE['alpha_i']}, {BASELINE['beta_i']}), "
          f"E[X]={BASELINE['alpha_i']/BASELINE['beta_i']:.0f}\n")

    table1 = {}

    # lambda_i in {0.001, ..., 0.008}
    print("[1/3] Sensitivity: lambda_i")
    out = run_sensitivity(
        param_name=r'\lambda_i',
        param_label=r'Risk Aversion $\lambda_i$',
        param_values=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
        override_key='lam_i',
        cmap_name='viridis',
        filename='asym_sensitivity_lambda.png',
    )
    if out: table1['lambda_i'] = out['summary']

    # a_i in {1.25, ..., 2.25}
    print("\n[2/3] Sensitivity: a_i")
    out = run_sensitivity(
        param_name='a_i',
        param_label=r'Price Sensitivity $a_i$',
        param_values=[1.25, 1.5, 1.7834, 2.0, 2.25],
        override_key='ai',
        cmap_name='plasma',
        filename='asym_sensitivity_a.png',
    )
    if out: table1['a_i'] = out['summary']

    # e_i in {0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20}
    print("\n[3/3] Sensitivity: e_i")
    out = run_sensitivity(
        param_name='e_i',
        param_label=r'Expense Rate $e_i$',
        param_values=[0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
        override_key='ei',
        cmap_name='copper',
        filename='asym_sensitivity_e.png',
    )
    if out: table1['e_i'] = out['summary']

    # Consolidated Table 2 directional summary
    print("\n" + "=" * 65)
    print(f"{'TABLE 2: Directional Effects (player i parameter up)':^65}")
    print("=" * 65)
    print(f"  {'Param up':<12}{'Mean p_i*':<14}{'Mean p_j*':<14}{'Amplitude(i)':<16}")
    print("  " + "-" * 54)
    label_map = {'lambda_i': 'lambda_i', 'a_i': 'a_i', 'e_i': 'e_i'}
    short = lambda s: ('up' if 'up' in s else
                       'down' if 'down' in s else
                       'flat' if 'flat' in s else 'n.m.')
    for key in ['lambda_i', 'a_i', 'e_i']:
        if key in table1:
            s = table1[key]
            print(f"  {label_map[key]:<12}"
                  f"{short(s['mean_i']):<14}"
                  f"{short(s['mean_j']):<14}"
                  f"{short(s['amp_i']):<16}")
    print("=" * 65)
    print("  Figures: asym_sensitivity_lambda.png, "
          "asym_sensitivity_a.png, asym_sensitivity_e.png")


if __name__ == "__main__":
    main()
