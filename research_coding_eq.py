# -*- coding: utf-8 -*-
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from typing import List, Tuple, Optional

PLOT_DETAILED = False


# Following part is about Utility Functions and Parameters

def get_gamma_log_mgf(lam: float, alpha: float, beta: float) -> float:
    """
    Log-MGF of Gamma(alpha, beta) distribution at point lam.
    M_X(lam) = (1 - lam/beta)^{-alpha}
    log M_X(lam) = -alpha * log(1 - lam/beta)
    Requires lam < beta (Risk Integrability Assumption).
    """
    if lam >= beta:
        raise ValueError(
            f"Risk Integrability violated: lambda={lam} >= beta={beta}. "
            f"Need lambda < beta for finite MGF."
        )
    return -alpha * math.log(1.0 - lam / beta)


def get_user_input(prompt: str, default) -> float:
    """Get user input with a default value."""
    val_str = input(f"  {prompt} [Default: {default}]: ")
    return float(val_str) if val_str.strip() else float(default)


class GameParameters:
    """Container for all model parameters."""

    def __init__(self):
        # t
        self.N: int = 5

        self.p_min: float = 1e-3
        self.p_max: float = 500.0

        # n0
        self.ni0: float = 2500.0
        self.nj0: float = 2500.0

        # e
        self.ei: float = 0.07
        self.ej: float = 0.07

        # lam
        self.lam_i: float = 0.004
        self.lam_j: float = 0.004

        # a
        self.ai: float = 1.7834
        self.aj: float = 1.7834

        # w0
        self.wi0: float = 0.0
        self.wj0: float = 0.0

        # Gamma distribution
        self.alpha_i: float = 3.0
        self.beta_i: float = 0.03
        self.alpha_j: float = 3.0
        self.beta_j: float = 0.03

        # some middle figures
        self.kappa_i: float = 0.0   # lambda_i * (1 - e_i)
        self.kappa_j: float = 0.0
        self.log_mgf_i: float = 0.0  # log M_{X_i}(lambda_i)
        self.log_mgf_j: float = 0.0

    def precompute(self):
        if not isinstance(self.N, int) or self.N < 1:
            raise ValueError(f"T must be a positive integer; got {self.N!r}.")
        if not (0.0 < self.p_min < self.p_max):
            raise ValueError(
                f"Need 0 < p_min < p_max; got [{self.p_min}, {self.p_max}]."
            )
        for name, value in (("e_i", self.ei), ("e_j", self.ej)):
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must lie in [0, 1); got {value}.")
        for name, value in (("lambda_i", self.lam_i), ("lambda_j", self.lam_j)):
            if value <= 0.0:
                raise ValueError(f"{name} must be strictly positive; got {value}.")
        for name, value in (("a_i", self.ai), ("a_j", self.aj)):
            if value <= 0.0:
                raise ValueError(f"{name} must be strictly positive; got {value}.")
        for name, value in (
            ("alpha_i", self.alpha_i), ("beta_i", self.beta_i),
            ("alpha_j", self.alpha_j), ("beta_j", self.beta_j),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be strictly positive; got {value}.")
        for name, value in (("n_i0", self.ni0), ("n_j0", self.nj0)):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative; got {value}.")

        self.kappa_i = self.lam_i * (1.0 - self.ei)
        self.kappa_j = self.lam_j * (1.0 - self.ej)
        self.log_mgf_i = get_gamma_log_mgf(self.lam_i, self.alpha_i, self.beta_i)
        self.log_mgf_j = get_gamma_log_mgf(self.lam_j, self.alpha_j, self.beta_j)

    @property
    def breakeven_i(self) -> float:
        return self.log_mgf_i / self.kappa_i if self.kappa_i > 0 else float('inf')

    @property
    def breakeven_j(self) -> float:
        return self.log_mgf_j / self.kappa_j if self.kappa_j > 0 else float('inf')

    def print_summary(self):
        print(f"\n  {'Model Parameters':=^60}")
        print(f"  Periods T          = {self.N}")
        print(f"  Premium set        = [{self.p_min}, {self.p_max}]")
        print(f"  n_{{i,0}}={self.ni0:.0f},  n_{{j,0}}={self.nj0:.0f}")
        print(f"  e_i={self.ei},  e_j={self.ej}")
        print(f"  lambda_i={self.lam_i},  lambda_j={self.lam_j}")
        print(f"  a_i={self.ai},  a_j={self.aj}")
        print(f"  X_i ~ Gamma({self.alpha_i}, {self.beta_i}), E[X_i]={self.alpha_i / self.beta_i:.2f}")
        print(f"  X_j ~ Gamma({self.alpha_j}, {self.beta_j}), E[X_j]={self.alpha_j / self.beta_j:.2f}")
        print(f"  kappa_i = {self.kappa_i:.6f},  kappa_j = {self.kappa_j:.6f}")
        print(f"  log M_Xi(lambda_i) = {self.log_mgf_i:.6f}")
        print(f"  log M_Xj(lambda_j) = {self.log_mgf_j:.6f}")
        print(f"  Break-even premium i = {self.breakeven_i:.4f}")
        print(f"  Break-even premium j = {self.breakeven_j:.4f}")
        print(f"  {'':=^60}")


def make_params(T, ni0, nj0, ei, ej, lam_i, lam_j, ai, aj,
                alpha_i=3.0, beta_i=0.03, alpha_j=3.0, beta_j=0.03,
                wi0=0.0, wj0=0.0,
                p_min=1e-3, p_max=500.0) -> GameParameters:
    p = GameParameters()
    p.N = T
    p.ni0, p.nj0 = ni0, nj0
    p.ei, p.ej = ei, ej
    p.lam_i, p.lam_j = lam_i, lam_j
    p.ai, p.aj = ai, aj
    p.alpha_i, p.beta_i = alpha_i, beta_i
    p.alpha_j, p.beta_j = alpha_j, beta_j
    p.wi0, p.wj0 = wi0, wj0
    p.p_min, p.p_max = p_min, p_max
    p.precompute()
    return p


# Following part is about core model functions (equations in the paper)

def f_func(p_own: float, p_opp: float, a: float) -> float:
    """
    (Eq. 5): f_i(p_i, p_j) = exp(-a_i * (p_i - p_j) / p_j)
    """
    if p_own <= 0.0 or p_opp <= 0.0:
        raise ValueError(
            f"Premiums must be strictly positive; got own={p_own}, opponent={p_opp}."
        )
    return math.exp(-a * (p_own - p_opp) / p_opp)


def log_C_func(p: float, kappa: float, log_mgf: float) -> float:
    """
    (Eq. 10): log C_i(p) = -kappa_i * p + log M_{X_i}(lambda_i)
    """
    return -kappa * p + log_mgf


def Z_step(p_own: float, p_opp: float, Z_future: float,
           a: float, kappa: float, log_mgf: float) -> float:
    """
    (Eq. 15 / Eq. 39): Z_t = f_i(p_i, p_j) * (C_i(p_i) * exp(Z_{t+1}) - 1)
    """
    f = f_func(p_own, p_opp, a)
    exponent = log_C_func(p_own, kappa, log_mgf) + Z_future
    if exponent > 700.0:
        raise FloatingPointError(
            f"Z recursion overflow: exp argument is {exponent:.6g}. "
            "Check the parameter range and starting premium path."
        )
    return f * math.expm1(exponent)



# Following part is about closed-loop solver, using backward induction


def analytical_best_response(p_opp: float, Z_future: float,
                              kappa: float, log_mgf: float, a: float,
                              p_min: float = 1e-3,
                              p_max: float = 500.0) -> float:
    """
    C_i(p_i^*) * exp(Z) = a_i / (a_i + kappa_i * p_j)
    p_i^* = (1/kappa_i) * [log_mgf + Z + log(1 + kappa_i * p_j / a_i)]
    """
    if p_opp <= 0.0 or kappa <= 0.0 or a <= 0.0:
        raise ValueError(
            f"Need p_opp, kappa, and a > 0; got {p_opp}, {kappa}, {a}."
        )
    val = (log_mgf + Z_future + math.log1p(kappa * p_opp / a)) / kappa
    return float(np.clip(val, p_min, p_max))


def solve_stage_nash(Z_future_i: float, Z_future_j: float,
                     params: GameParameters,
                     tol: float = 1e-14,
                     max_iter: int = 10000) -> Tuple[float, float]:
    # our initial guess is slightly above break-even
    pi = float(np.clip(
        (params.log_mgf_i + Z_future_i) / params.kappa_i * 1.3,
        params.p_min, params.p_max
    ))
    pj = float(np.clip(
        (params.log_mgf_j + Z_future_j) / params.kappa_j * 1.3,
        params.p_min, params.p_max
    ))

    for it in range(max_iter):
        prev_pi, prev_pj = pi, pj 
        pi_new = analytical_best_response(pj, Z_future_i,
                                           params.kappa_i, params.log_mgf_i, params.ai,
                                           params.p_min, params.p_max)
        pj_new = analytical_best_response(pi_new, Z_future_j,
                                           params.kappa_j, params.log_mgf_j, params.aj,
                                           params.p_min, params.p_max)

        if abs(pi_new - pi) < tol and abs(pj_new - pj) < tol:
            return pi_new, pj_new

        pi, pj = pi_new, pj_new

    raise RuntimeError(
        f"Stage Nash iteration did not converge after {max_iter} iterations "
        f"(diff_i={abs(pi - prev_pi):.2e}, diff_j={abs(pj - prev_pj):.2e})."
    )


def solve_closed_loop(params: GameParameters, verbose: bool = True):
    N = params.N
    if verbose:
        print(f"\n{'=' * 65}")
        print(f"  Solving {N}-Period Closed-Loop Equilibrium (Backward Induction)")
        print(f"{'=' * 65}")

    prem_i = [0.0] * N
    prem_j = [0.0] * N
    # Z_i[k] stores Z_{i, k+1} 
    Z_i = [0.0] * (N + 1)
    Z_j = [0.0] * (N + 1)

    for t in range(N, 0, -1):
        idx = t - 1       
        Zf_i = Z_i[t]     
        Zf_j = Z_j[t]     
        # solve static Nash game
        pi_star, pj_star = solve_stage_nash(Zf_i, Zf_j, params)
        prem_i[idx] = pi_star
        prem_j[idx] = pj_star
        # update Z in Eq. 39
        Z_i[idx] = Z_step(pi_star, pj_star, Zf_i,
                          params.ai, params.kappa_i, params.log_mgf_i)
        Z_j[idx] = Z_step(pj_star, pi_star, Zf_j,
                          params.aj, params.kappa_j, params.log_mgf_j)

        if verbose:
            print(f"  Period {t:2d}: p_i*={pi_star:10.4f}  p_j*={pj_star:10.4f}  |  "
                  f"Z_i={Z_i[idx]:12.8f}  Z_j={Z_j[idx]:12.8f}")

    if verbose:
        print("  CL Backward Induction Complete.")
    return prem_i, prem_j, Z_i, Z_j


# Following part is about open-Loop solver

def compute_Z1_and_gradient(
    p_own: np.ndarray, p_opp: np.ndarray,
    a: float, kappa: float, log_mgf: float, N: int
) -> Tuple[float, np.ndarray]:
    """
    Compute Z_{i,1} and its gradient Uses the backward recursion (Eq. 15): Z_t = f(p_t, q_t) * (exp(log_C(p_t) + Z_{t+1}) - 1)
    Gradient via chain rule: dZ_1/dp_s = [prod_{r=0}^{s-1} f_r * C_r * exp(Z_{r+1})] * dZ_s/dp_s and each factor dZ_r/dZ_{r+1} = f_r * C_r * exp(Z_{r+1}) > 0.
    """
    # compute all Z values (backward)
    Z = np.zeros(N + 1)   # Z[N] = Z_{T+1} = 0
    for t in range(N - 1, -1, -1):
        p, q = p_own[t], p_opp[t]
        Z[t] = Z_step(p, q, Z[t + 1], a, kappa, log_mgf)

    # compute gradient (forward)
    grad = np.zeros(N)
    chain_factor = 1.0   # product of dZ_r/dZ_{r+1} from r=0 to s-1

    for s in range(N):
        p, q = p_own[s], p_opp[s]
        f = f_func(p, q, a)
        exponent = -kappa * p + log_mgf + Z[s + 1]
        if exponent > 700.0:
            raise FloatingPointError(
                f"Gradient overflow: exp argument is {exponent:.6g}."
            )
        C_eZ = math.exp(exponent)

        # dZ_s/dp_s = df/dp * (C*eZ - 1) + f * dC/dp * eZ
        df_dp = f * (-a / q)
        dZ_s_dp_s = df_dp * math.expm1(exponent) - f * kappa * C_eZ

        grad[s] = chain_factor * dZ_s_dp_s

        # dZ_s/dZ_{s+1} = f * C * eZ
        chain_factor *= f * C_eZ

    return Z[0], grad


def verify_gradient(p_own, p_opp, a, kappa, log_mgf, N,
                    abs_tol=1e-8):
    """Verify the analytical gradient using adaptive centered differences."""
    _, grad_a = compute_Z1_and_gradient(p_own, p_opp, a, kappa, log_mgf, N)
    grad_n = np.zeros(N)
    base_step = np.cbrt(np.finfo(float).eps)
    for i in range(N):
        p_plus = p_own.copy()
        p_minus = p_own.copy()
        step = base_step * max(1.0, abs(p_own[i]))
        step = min(step, 0.49 * p_own[i])
        p_plus[i] += step
        p_minus[i] -= step
        Z1_plus, _ = compute_Z1_and_gradient(p_plus, p_opp, a, kappa, log_mgf, N)
        Z1_minus, _ = compute_Z1_and_gradient(p_minus, p_opp, a, kappa, log_mgf, N)
        grad_n[i] = (Z1_plus - Z1_minus) / (2.0 * step)

    print("\n  Gradient Verification (Player i):")
    print(f"  {'Period':<8} {'Analytical':<16} {'Centered FD':<16} "
          f"{'Abs.Error':<14} {'Rel.Error':<14}")
    max_abs_err = 0.0
    for i in range(N):
        abs_err = abs(grad_a[i] - grad_n[i])
        denom = max(abs(grad_a[i]), abs(grad_n[i]), 1e-12)
        rel_err = abs_err / denom
        max_abs_err = max(max_abs_err, abs_err)
        print(f"  {i+1:<8} {grad_a[i]:<16.8e} {grad_n[i]:<16.8e} "
              f"{abs_err:<14.2e} {rel_err:<14.2e}")

    print(f"  Max absolute gradient error = {max_abs_err:.2e}")
    if max_abs_err > abs_tol:
        raise RuntimeError(
            f"Analytical gradient check failed: {max_abs_err:.3e} > {abs_tol:.3e}."
        )
    return max_abs_err


def compute_Z_paths(pi: np.ndarray, pj: np.ndarray,
                    params: GameParameters) -> Tuple[np.ndarray, np.ndarray]:
    """Compute both continuation-scalar paths for a joint premium path."""
    N = params.N
    if pi.shape != (N,) or pj.shape != (N,):
        raise ValueError(f"Expected two premium vectors of length {N}.")

    Z_i = np.zeros(N + 1)
    Z_j = np.zeros(N + 1)
    for t in range(N - 1, -1, -1):
        Z_i[t] = Z_step(
            pi[t], pj[t], Z_i[t + 1],
            params.ai, params.kappa_i, params.log_mgf_i
        )
        Z_j[t] = Z_step(
            pj[t], pi[t], Z_j[t + 1],
            params.aj, params.kappa_j, params.log_mgf_j
        )
    return Z_i, Z_j


def open_loop_projected_residual(x: np.ndarray,
                                 params: GameParameters) -> np.ndarray:
    """
    Return the 2T projected best-response residuals for the OL equilibrium.

    This formulation remains valid at a premium bound. For an interior solution,
    it reduces exactly to the open-loop first-order conditions in Eq. (58).
    """
    N = params.N
    x = np.asarray(x, dtype=float)
    if x.shape != (2 * N,):
        raise ValueError(f"Expected a vector of length {2 * N}; got {x.shape}.")

    pi = x[:N]
    pj = x[N:]
    Z_i, Z_j = compute_Z_paths(pi, pj, params)

    br_i = (
        params.log_mgf_i
        + Z_i[1:]
        + np.log1p(params.kappa_i * pj / params.ai)
    ) / params.kappa_i
    br_j = (
        params.log_mgf_j
        + Z_j[1:]
        + np.log1p(params.kappa_j * pi / params.aj)
    ) / params.kappa_j

    br_i = np.clip(br_i, params.p_min, params.p_max)
    br_j = np.clip(br_j, params.p_min, params.p_max)
    return np.concatenate((pi - br_i, pj - br_j))


def solve_open_loop(params: GameParameters,
                    init_guess_i: Optional[np.ndarray] = None,
                    init_guess_j: Optional[np.ndarray] = None,
                    verbose: bool = True,
                    residual_tol: float = 1e-10):
    """
    Independently solve the full 2T-dimensional OL equilibrium system.

    The solver starts from a neutral or user-supplied path and never reads the
    closed-loop solution. SciPy estimates the Jacobian numerically, providing a
    computational check separate from the analytical-gradient implementation.
    """
    N = params.N
    if verbose:
        print(f"\n{'=' * 65}")
        print(f"  Solving {N}-Period Open-Loop Nash Equilibrium Independently")
        print(f"{'=' * 65}")

    supplied_i = init_guess_i is not None
    supplied_j = init_guess_j is not None
    if supplied_i != supplied_j:
        raise ValueError("Provide both initial premium paths or neither one.")

    if supplied_i:
        pi0 = np.asarray(init_guess_i, dtype=float)
        pj0 = np.asarray(init_guess_j, dtype=float)
        if pi0.shape != (N,) or pj0.shape != (N,):
            raise ValueError(f"Initial premium paths must both have length {N}.")
    else:
        pi0 = np.full(N, 1.5 * params.breakeven_i)
        pj0 = np.full(N, 1.5 * params.breakeven_j)

    x0 = np.clip(
        np.concatenate((pi0, pj0)), params.p_min, params.p_max
    )
    result = least_squares(
        open_loop_projected_residual,
        x0,
        args=(params,),
        bounds=(params.p_min, params.p_max),
        method="trf",
        x_scale="jac",
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        max_nfev=5000,
    )

    # Bounded least-squares keeps active variables infinitesimally inside a
    # bound. A short projected fixed-point polish snaps genuine boundary
    # solutions to the exact bound and removes the resulting artificial
    # residual without using any closed-loop information.
    solution = result.x.copy()
    polish_iterations = 0
    for polish_iterations in range(100):
        residual = open_loop_projected_residual(solution, params)
        polished = np.clip(
            solution - residual, params.p_min, params.p_max
        )
        if np.linalg.norm(polished - solution, ord=np.inf) <= 1e-13:
            solution = polished
            break
        solution = polished

    residual = open_loop_projected_residual(solution, params)
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    if not result.success or not np.all(np.isfinite(solution)):
        raise RuntimeError(
            f"Independent OL solver failed: success={result.success}; "
            f"message={result.message}"
        )
    if residual_inf > residual_tol:
        raise RuntimeError(
            f"Independent OL residual {residual_inf:.3e} exceeds "
            f"the required tolerance {residual_tol:.3e}."
        )

    pi = solution[:N]
    pj = solution[N:]
    Z_i, Z_j = compute_Z_paths(pi, pj, params)

    if verbose:
        print(f"  Solver evaluations = {result.nfev}")
        print(f"  Projected polish iterations = {polish_iterations + 1}")
        print(f"  Max projected-BR residual = {residual_inf:.3e}")
        for t in range(N):
            print(f"  Period {t+1:2d}: p_i*={pi[t]:10.4f}  p_j*={pj[t]:10.4f}  |  "
                  f"Z_i={Z_i[t]:12.8f}  Z_j={Z_j[t]:12.8f}")

    return pi.tolist(), pj.tolist(), Z_i.tolist(), Z_j.tolist()


def verify_open_loop_multistart(params: GameParameters,
                                reference_i, reference_j,
                                n_random: int = 5,
                                seed: int = 2026,
                                agreement_tol: float = 1e-9) -> float:
    """Re-solve OL from independent seeded random paths and compare solutions."""
    if n_random < 1:
        raise ValueError("n_random must be at least one.")

    N = params.N
    reference = np.concatenate((reference_i, reference_j)).astype(float)
    rng = np.random.default_rng(seed)
    low_i = max(params.p_min, 0.5 * params.breakeven_i)
    high_i = min(params.p_max, 2.5 * params.breakeven_i)
    low_j = max(params.p_min, 0.5 * params.breakeven_j)
    high_j = min(params.p_max, 2.5 * params.breakeven_j)

    max_difference = 0.0
    for _ in range(n_random):
        start_i = rng.uniform(low_i, high_i, N)
        start_j = rng.uniform(low_j, high_j, N)
        pi, pj, _, _ = solve_open_loop(
            params, start_i, start_j, verbose=False
        )
        solution = np.concatenate((pi, pj))
        max_difference = max(
            max_difference, float(np.max(np.abs(solution - reference)))
        )

    print(f"  OL multi-start check ({n_random} random paths): "
          f"max solution difference = {max_difference:.3e}")
    if max_difference > agreement_tol:
        raise RuntimeError(
            f"OL multi-start solutions differ by {max_difference:.3e}, above "
            f"the tolerance {agreement_tol:.3e}."
        )
    return max_difference

# Metrics Computation

def compute_loss(Z_1: float, lam: float, w0: float, n0: float) -> float:
    """Loss function (Eq. 18): L_i = -lambda_i * W_{i,0} + n_{i,0} * Z_{i,1}"""
    return -lam * w0 + n0 * Z_1


def compute_expected_utility(loss: float) -> float:
    """Expected utility: J_i = -exp(L_i)"""
    if loss > 700:
        return -1e300
    if loss < -700:
        return -0.0  # effectively 0
    return -math.exp(loss)


def compute_all_metrics(prem_i, prem_j, Z_i, Z_j, params, label="", verbose=True):
    """Compute and print all relevant metrics for one equilibrium."""
    N = params.N

    L_i = compute_loss(Z_i[0], params.lam_i, params.wi0, params.ni0)
    L_j = compute_loss(Z_j[0], params.lam_j, params.wj0, params.nj0)
    J_i = compute_expected_utility(L_i)
    J_j = compute_expected_utility(L_j)

    # Expected portfolio sizes along equilibrium path
    en_i = [params.ni0]
    en_j = [params.nj0]
    for t in range(N):
        fi = f_func(prem_i[t], prem_j[t], params.ai)
        fj = f_func(prem_j[t], prem_i[t], params.aj)
        en_i.append(en_i[-1] * fi)
        en_j.append(en_j[-1] * fj)

    # Expected per-period surplus: E[S_{i,t}] = n_{i,t-1} * f_i * [(1-e_i)*p_i - E[X_i]]
    mean_Xi = params.alpha_i / params.beta_i
    mean_Xj = params.alpha_j / params.beta_j
    surplus_i = []
    surplus_j = []
    for t in range(N):
        fi = f_func(prem_i[t], prem_j[t], params.ai)
        fj = f_func(prem_j[t], prem_i[t], params.aj)
        expected_n_i = en_i[t] * fi  # = en_i[t+1]
        expected_n_j = en_j[t] * fj
        s_i = expected_n_i * ((1.0 - params.ei) * prem_i[t] - mean_Xi)
        s_j = expected_n_j * ((1.0 - params.ej) * prem_j[t] - mean_Xj)
        surplus_i.append(s_i)
        surplus_j.append(s_j)

    if verbose:
        print(f"\n  [{label}] Key Metrics:")
        print(f"    Z_{{i,1}} = {Z_i[0]:.10f}")
        print(f"    Z_{{j,1}} = {Z_j[0]:.10f}")
        print(f"    Loss_i  = {L_i:.6f}")
        print(f"    Loss_j  = {L_j:.6f}")
        print(f"    J_i     = -exp({L_i:.2f})")
        print(f"    J_j     = -exp({L_j:.2f})")
        print(f"    Mean premium i = {np.mean(prem_i):.4f}")
        print(f"    Mean premium j = {np.mean(prem_j):.4f}")

    return {
        'L_i': L_i, 'L_j': L_j, 'J_i': J_i, 'J_j': J_j,
        'en_i': en_i, 'en_j': en_j,
        'surplus_i': surplus_i, 'surplus_j': surplus_j,
    }

# Visualization

def plot_comprehensive_results(
    ol_i, ol_j, cl_i, cl_j,
    ol_Z_i, ol_Z_j, cl_Z_i, cl_Z_j,
    ol_metrics, cl_metrics,
    params, savename='equilibrium_results.png'
):
    """Generate the full 9-panel comparison figure (kept for completeness)."""
    N = params.N
    periods = list(range(1, N + 1))
    z_periods = list(range(1, N + 2))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"Insurance Duopoly Equilibrium: Open-Loop vs Closed-Loop (T={N})",
        fontsize=16, fontweight='bold', y=0.98
    )

    c_ol_i = '#2196F3'   # blue
    c_cl_i = '#F44336'   # red
    c_ol_j = '#4CAF50'   # green
    c_cl_j = '#FF9800'   # orange

    # Player I's Premium Path
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(periods, ol_i, 'o--', label='Open-Loop', color=c_ol_i, lw=2, ms=7, zorder=5)
    ax1.plot(periods, cl_i, 's-', label='Closed-Loop', color=c_cl_i, lw=2, ms=7, zorder=4)
    ax1.axhline(y=params.breakeven_i, color='gray', ls=':', lw=1,
                label=f'Break-even={params.breakeven_i:.1f}')
    for t in range(N):
        ax1.annotate(f"{cl_i[t]:.1f}", (periods[t], cl_i[t]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=7, color=c_cl_i, fontweight='bold')
    ax1.set_title("Player I Premium Path", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Period")
    ax1.set_ylabel("Premium (p_i)")
    ax1.set_xticks(periods)
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)

    # Player J's Premium Path
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(periods, ol_j, 'o--', label='Open-Loop', color=c_ol_j, lw=2, ms=7, zorder=5)
    ax2.plot(periods, cl_j, 's-', label='Closed-Loop', color=c_cl_j, lw=2, ms=7, zorder=4)
    ax2.axhline(y=params.breakeven_j, color='gray', ls=':', lw=1,
                label=f'Break-even={params.breakeven_j:.1f}')
    for t in range(N):
        ax2.annotate(f"{cl_j[t]:.1f}", (periods[t], cl_j[t]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=7, color=c_cl_j, fontweight='bold')
    ax2.set_title("Player J Premium Path", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Premium (p_j)")
    ax2.set_xticks(periods)
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)

    # Premium Difference (CL - OL)
    ax3 = fig.add_subplot(3, 3, 3)
    diff_i = [cl_i[t] - ol_i[t] for t in range(N)]
    diff_j = [cl_j[t] - ol_j[t] for t in range(N)]
    width = 0.35
    x = np.array(periods)
    ax3.bar(x - width / 2, diff_i, width, label='Player I', color=c_ol_i, alpha=0.7)
    ax3.bar(x + width / 2, diff_j, width, label='Player J', color=c_ol_j, alpha=0.7)
    ax3.axhline(y=0, color='black', lw=0.8)
    max_diff = max(max(abs(d) for d in diff_i), max(abs(d) for d in diff_j), 1e-10)
    ax3.set_title(f"Premium Diff (CL-OL)\nMax |Δ|={max_diff:.2e}", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Δ Premium")
    ax3.set_xticks(periods)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Z Values Player I
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(z_periods, ol_Z_i[:N + 1], 'o--', label='OL Z_i', color=c_ol_i, lw=2, ms=6)
    ax4.plot(z_periods, cl_Z_i[:N + 1], 's-', label='CL Z_i', color=c_cl_i, lw=2, ms=6)
    ax4.axhline(y=0, color='black', lw=0.8, ls='--')
    for t in range(N + 1):
        ax4.annotate(f"{cl_Z_i[t]:.4f}", (z_periods[t], cl_Z_i[t]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=6, color=c_cl_i)
    ax4.set_title("Z Values (Player I)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Period t")
    ax4.set_ylabel("Z_{i,t}")
    ax4.set_xticks(z_periods)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Z Values Player J
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(z_periods, ol_Z_j[:N + 1], 'o--', label='OL Z_j', color=c_ol_j, lw=2, ms=6)
    ax5.plot(z_periods, cl_Z_j[:N + 1], 's-', label='CL Z_j', color=c_cl_j, lw=2, ms=6)
    ax5.axhline(y=0, color='black', lw=0.8, ls='--')
    for t in range(N + 1):
        ax5.annotate(f"{cl_Z_j[t]:.4f}", (z_periods[t], cl_Z_j[t]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=6, color=c_cl_j)
    ax5.set_title("Z Values (Player J)", fontsize=12, fontweight='bold')
    ax5.set_xlabel("Period t")
    ax5.set_ylabel("Z_{j,t}")
    ax5.set_xticks(z_periods)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Both Players CL Equilibrium
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(periods, cl_i, 'o-', label='CL p_i', color=c_cl_i, lw=2.5, ms=8)
    ax6.plot(periods, cl_j, 's-', label='CL p_j', color=c_cl_j, lw=2.5, ms=8)
    ax6.axhline(y=params.breakeven_i, color=c_cl_i, ls=':', lw=1, alpha=0.5)
    ax6.axhline(y=params.breakeven_j, color=c_cl_j, ls=':', lw=1, alpha=0.5)
    ax6.fill_between(periods,
                     [min(cl_i[t], cl_j[t]) for t in range(N)],
                     [max(cl_i[t], cl_j[t]) for t in range(N)],
                     alpha=0.1, color='purple')
    ax6.set_title("CL Equilibrium: Both Players", fontsize=12, fontweight='bold')
    ax6.set_xlabel("Period")
    ax6.set_ylabel("Premium")
    ax6.set_xticks(periods)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Expected Portfolio Sizes
    ax7 = fig.add_subplot(3, 3, 7)
    port_periods = list(range(0, N + 1))
    ax7.plot(port_periods, cl_metrics['en_i'], 'o-', label='CL E[n_i]',
             color=c_cl_i, lw=2, ms=6)
    ax7.plot(port_periods, cl_metrics['en_j'], 's-', label='CL E[n_j]',
             color=c_cl_j, lw=2, ms=6)
    ax7.plot(port_periods, ol_metrics['en_i'], 'x--', label='OL E[n_i]',
             color=c_ol_i, lw=1.5, ms=6)
    ax7.set_title("Expected Portfolio Size", fontsize=12, fontweight='bold')
    ax7.set_xlabel("Period (end-of)")
    ax7.set_ylabel("E[n_i,t]")
    ax7.set_xticks(port_periods)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # Expected Surplus per Period
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.bar([p - 0.18 for p in periods], cl_metrics['surplus_i'], 0.35,
            label='CL E[S_i]', color=c_cl_i, alpha=0.7)
    ax8.bar([p + 0.18 for p in periods], cl_metrics['surplus_j'], 0.35,
            label='CL E[S_j]', color=c_cl_j, alpha=0.7)
    ax8.axhline(y=0, color='black', lw=0.8)
    ax8.set_title("Expected Surplus per Period", fontsize=12, fontweight='bold')
    ax8.set_xlabel("Period")
    ax8.set_ylabel("E[S_{i,t}]")
    ax8.set_xticks(periods)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # Summary
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = (
        f"Model Summary\n"
        f"{'─' * 40}\n"
        f"Periods: {N}\n"
        f"n_i0={params.ni0:.0f}, n_j0={params.nj0:.0f}\n"
        f"λ_i={params.lam_i}, λ_j={params.lam_j}\n"
        f"a_i={params.ai}, a_j={params.aj}\n"
        f"e_i={params.ei}, e_j={params.ej}\n"
        f"X_i~Ga({params.alpha_i},{params.beta_i}), "
        f"X_j~Ga({params.alpha_j},{params.beta_j})\n"
        f"{'─' * 40}\n"
        f"CL Loss_i = {cl_metrics['L_i']:.4f}\n"
        f"OL Loss_i = {ol_metrics['L_i']:.4f}\n"
        f"CL Loss_j = {cl_metrics['L_j']:.4f}\n"
        f"OL Loss_j = {ol_metrics['L_j']:.4f}\n"
        f"{'─' * 40}\n"
        f"Max |p_CL - p_OL| = {max_diff:.2e}\n"
        f"(OL=CL is expected by theory)"
    )
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    print(f"\n  Detailed plot saved to '{savename}'")
    plt.close(fig)


def plot_three_examples(results, savename="three_examples_premiums.png"):
    """
    Layout: 3 rows x 2 columns (portrait, height > width).
    Each row is one example; left column = Closed-Loop, right column = Open-Loop.
    CL and OL are computed by different algorithms (backward induction vs.
    full-path optimization); by Theorem 4.1 they coincide, which the
    side-by-side panels make visible.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 5.2 * n))
    if n == 1:
        axes = np.array([axes])
    fig.suptitle("Equilibrium Premium Paths: Closed-Loop vs Open-Loop",
                 fontsize=16, fontweight='bold')

    c_i = '#C62828'   # Player I (red)
    c_j = '#1565C0'   # Player J (blue)

    def draw_panel(ax, p, prem_i, prem_j, title):
        periods = list(range(1, p.N + 1))
        ax.plot(periods, prem_i, 'o-', color=c_i, lw=2.2, ms=8,
                label='Player I ($p_i$)', zorder=5)
        ax.plot(periods, prem_j, 's-', color=c_j, lw=2.2, ms=8,
                label='Player J ($p_j$)', zorder=4)
        ax.axhline(p.breakeven_i, color=c_i, ls=':', lw=1, alpha=0.4)
        ax.axhline(p.breakeven_j, color=c_j, ls=':', lw=1, alpha=0.4)
        for t in range(p.N):
            ax.annotate(f"{prem_i[t]:.0f}", (periods[t], prem_i[t]),
                        textcoords="offset points", xytext=(0, 11),
                        ha='center', fontsize=7.5, color=c_i, fontweight='bold')
            ax.annotate(f"{prem_j[t]:.0f}", (periods[t], prem_j[t]),
                        textcoords="offset points", xytext=(0, -15),
                        ha='center', fontsize=7.5, color=c_j, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel("Period $t$", fontsize=10)
        ax.set_ylabel("Premium", fontsize=10)
        ax.set_xticks(periods)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.margins(y=0.22)

    for row, r in enumerate(results):
        p = r['params']
        ax_cl = axes[row, 0]
        ax_ol = axes[row, 1]

        # Left column: Closed-Loop (backward induction)
        draw_panel(ax_cl, p, r['cl_i'], r['cl_j'],
                   f"{r['title']} — Closed-Loop  (T={p.N})")

        # Right column: Open-Loop (full-path optimization)
        draw_panel(ax_ol, p, r['ol_i'], r['ol_j'],
                   f"{r['title']} — Open-Loop  (T={p.N})")

    # Manual spacing: enlarge row gap (hspace) to stop x-labels colliding
    # with the next row's title; leave room for the suptitle (top).
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.07, right=0.97,
                        hspace=0.42, wspace=0.22)
    fig.savefig(savename, dpi=150, bbox_inches='tight')
    print(f"\n  Six-panel premium-path figure saved to '{savename}'")
    plt.show()


# FOC Verification

def verify_foc(prem_i, prem_j, Z_i, Z_j, params):
    """
    Verify that the first-order conditions are satisfied at the computed
    equilibrium.

    FOC for player i at period t:
      C_i * eZ = a_i / (a_i + kappa_i * p_j)
    """
    N = params.N
    print(f"\n  {'FOC Verification':=^60}")
    print(f"  {'Period':<8} {'LHS_i':<16} {'RHS_i':<16} {'AbsErr_i':<14} "
          f"{'LHS_j':<16} {'RHS_j':<16} {'AbsErr_j':<14}")

    max_err = 0.0
    for t in range(N):
        pi, pj = prem_i[t], prem_j[t]
        Zf_i = Z_i[t + 1]  # Z_{i,t+1}
        Zf_j = Z_j[t + 1]

        Ci = math.exp(-params.kappa_i * pi + params.log_mgf_i)
        lhs_i = Ci * math.exp(Zf_i)
        rhs_i = params.ai / (params.ai + params.kappa_i * pj)
        err_i = abs(lhs_i - rhs_i)

        Cj = math.exp(-params.kappa_j * pj + params.log_mgf_j)
        lhs_j = Cj * math.exp(Zf_j)
        rhs_j = params.aj / (params.aj + params.kappa_j * pi)
        err_j = abs(lhs_j - rhs_j)

        max_err = max(max_err, err_i, err_j)
        print(f"  {t+1:<8} {lhs_i:<16.10f} {rhs_i:<16.10f} {err_i:<14.2e} "
              f"{lhs_j:<16.10f} {rhs_j:<16.10f} {err_j:<14.2e}")

    print(f"  Max FOC error = {max_err:.2e}")
    return max_err


# Per-example full pipeline (all functionality retained)


def run_example(title: str, params: GameParameters):
    """
    Run the complete OL/CL solve + all verifications + metrics for one example.
    Returns a dict containing everything.
    """
    print("\n" + "#" * 90)
    print(f"  {title}")
    print("#" * 90)
    params.print_summary()

    t_start = time.time()

    # Closed-Loop (backward induction)
    cl_i, cl_j, cl_Zi, cl_Zj = solve_closed_loop(params)
    cl_metrics = compute_all_metrics(cl_i, cl_j, cl_Zi, cl_Zj, params, "Closed-Loop")

    # Open-Loop
    print("\n  [Open-Loop: independent neutral initial path]")
    ol_i, ol_j, ol_Zi, ol_Zj = solve_open_loop(params)
    ol_metrics = compute_all_metrics(ol_i, ol_j, ol_Zi, ol_Zj, params, "Open-Loop")

    # Re-solve OL from five seeded random paths
    multistart_difference = verify_open_loop_multistart(
        params, ol_i, ol_j, n_random=5
    )

    t_elapsed = time.time() - t_start

    # Gradient verification away from the stationary point. At the optimum,
    # both gradients are near zero and relative finite-difference errors are
    # dominated by roundoff rather than by formula accuracy.
    print("\n  Verifying analytical gradient at an off-equilibrium test path...")
    gradient_test_i = np.clip(
        np.asarray(ol_i) * (1.0 + np.linspace(-0.08, 0.08, params.N)),
        params.p_min, params.p_max
    )
    gradient_error = verify_gradient(
        gradient_test_i, np.array(ol_j),
                    params.ai, params.kappa_i, params.log_mgf_i, params.N)

    # FOC verification
    print("\n  Verifying FOCs at CL equilibrium:")
    verify_foc(cl_i, cl_j, cl_Zi, cl_Zj, params)
    print("\n  Verifying FOCs at OL equilibrium:")
    verify_foc(ol_i, ol_j, ol_Zi, ol_Zj, params)

    # Comparison table
    print(f"\n{'=' * 90}")
    print(f"{'FINAL RESULTS COMPARISON: ' + title:^90}")
    print(f"{'=' * 90}")
    print(f"{'Period':<8} | {'OL p_i':<12} {'CL p_i':<12} {'Diff_i':<14} | "
          f"{'OL p_j':<12} {'CL p_j':<12} {'Diff_j':<14}")
    print("-" * 90)
    for t in range(params.N):
        di = cl_i[t] - ol_i[t]
        dj = cl_j[t] - ol_j[t]
        print(f"{t + 1:<8} | {ol_i[t]:<12.6f} {cl_i[t]:<12.6f} {di:<14.2e} | "
              f"{ol_j[t]:<12.6f} {cl_j[t]:<12.6f} {dj:<14.2e}")
    print("-" * 90)
    print(f"  CL Loss_i = {cl_metrics['L_i']:.8f}    OL Loss_i = {ol_metrics['L_i']:.8f}    "
          f"Diff = {abs(cl_metrics['L_i'] - ol_metrics['L_i']):.2e}")
    print(f"  CL Loss_j = {cl_metrics['L_j']:.8f}    OL Loss_j = {ol_metrics['L_j']:.8f}    "
          f"Diff = {abs(cl_metrics['L_j'] - ol_metrics['L_j']):.2e}")

    max_cl_ol_difference = max(
        max(abs(ol_i[t] - cl_i[t]) for t in range(params.N)),
        max(abs(ol_j[t] - cl_j[t]) for t in range(params.N))
    )
    print(f"\n  Independent OL vs CL: Max |Δp| = {max_cl_ol_difference:.2e}")
    if max_cl_ol_difference > 1e-10:
        raise RuntimeError(
            f"Independent OL and CL differ by {max_cl_ol_difference:.3e}, "
            "which exceeds 1e-10."
        )
    print(f"  Computation time: {t_elapsed:.4f} seconds")

    # Optional detailed 9-panel figure (functionality retained)
    if PLOT_DETAILED:
        safe = title.split(':')[0].strip().replace(' ', '_')
        plot_comprehensive_results(
            ol_i, ol_j, cl_i, cl_j,
            ol_Zi, ol_Zj, cl_Zi, cl_Zj,
            ol_metrics, cl_metrics, params,
            savename=f"detailed_{safe}.png"
        )

    return {
        'title': title, 'params': params,
        'cl_i': cl_i, 'cl_j': cl_j,
        'ol_i': ol_i, 'ol_j': ol_j,
        'cl_metrics': cl_metrics, 'ol_metrics': ol_metrics,
        'max_cl_ol_difference': max_cl_ol_difference,
        'multistart_difference': multistart_difference,
        'gradient_error': gradient_error,
    }



# Main Program


def main():
    print("=" * 90)
    print("  Dynamic Competitive Insurance Pricing: OL vs CL Equilibrium")
    print("  Model from Zijie Wang (2026) — three-example batch run")
    print("=" * 90)

    # three examples from the paper
    examples = [
        ("Example 1: Symmetric",
         make_params(T=5, ni0=2500, nj0=1800,
                     ei=0.07, ej=0.07,
                     lam_i=0.004, lam_j=0.004,
                     ai=1.7834, aj=1.7834)),
        ("Example 2: Fully asymmetric",
         make_params(T=6, ni0=2500, nj0=1800,
                     ei=0.07, ej=0.12,
                     lam_i=0.003, lam_j=0.0009,
                     ai=1.7834, aj=2.0024)),
        ("Example 3: Asymmetric $\\lambda$, $e$",
         make_params(T=5, ni0=2500, nj0=2500,
                     ei=0.07, ej=0.12,
                     lam_i=0.003, lam_j=0.0009,
                     ai=1.7834, aj=1.7834)),
    ]

    results = []
    for title, params in examples:
        results.append(run_example(title, params))

    # New output format: three-panel premium-path figure 
    print("\n  Generating combined three-example premium-path figure...")
    plot_three_examples(results)


if __name__ == "__main__":
    main()
