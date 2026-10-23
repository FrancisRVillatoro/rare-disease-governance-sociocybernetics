#!/usr/bin/env python3
"""Rare Disease Complex: stylised sequential iterative-best-response DMPC pilot.

This module implements the R2/R3-corrected numerical pilot for the Kybernetes
manuscript. It is a *data-informed stylised computational pilot*, not an
empirically calibrated policy controller.

Key modelling choices
---------------------
* Eight normalised state components, including transversal organisational state O.
* Six actors own disjoint intervention channels H, R, HTA, B, V, O.
* Each actor optimises only its own finite-horizon sequence and treats the other
  actors' sequences as shared forecasts.
* Forecasts are exchanged sequentially (Gauss-Seidel / iterative best response).
* The initial empirical-anchor vector X0 is an exact autonomous equilibrium.
* Nonlinear reinforcement is centred at X0 and the organisational coupling scale
  is chosen to give a local autonomous stability margin in the nominal pilot.
* Equity is soft-constrained through explicit nonnegative feasibility slacks;
  the code eliminates the slacks analytically by substituting their optimal
  positive-part values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence
import numpy as np
from scipy.optimize import minimize

STATE_NAMES = ("M", "S", "P", "L", "E", "V", "C", "O")
ACTOR_NAMES = ("H", "R", "HTA", "B", "V", "O")
N_X = 8
N_U = 6


@dataclass(frozen=True)
class ModelParams:
    x0: np.ndarray
    xstar: np.ndarray
    Lambda: np.ndarray
    Ac: np.ndarray
    Bu: np.ndarray
    F: np.ndarray
    delta: float
    theta: np.ndarray
    kappa: float
    coupling_scale: float
    q_diags: np.ndarray
    r: np.ndarray
    u_max: float
    shared_budget: float
    a_min: float
    phi_max: float
    rho_a: float
    rho_phi: float
    status_quo_control: np.ndarray


@dataclass(frozen=True)
class SolverSettings:
    horizon: int = 4
    periods: int = 36
    max_sweeps: int = 20
    coordination_tol: float = 1.0e-5
    slsqp_maxiter: int = 100
    slsqp_ftol: float = 1.0e-9
    random_seed: int = 20260904


# Heterogeneous empirical anchors used only to initialise the stylised state.
X0 = np.array([0.213, 0.115, 0.405, 0.698, 0.540, 0.193, 0.172, 0.646], dtype=float)

# Normative analytical benchmark, not a shared institutional utility.
XSTAR = np.array([0.55, 0.36, 0.55, 0.74, 0.62, 0.24, 0.24, 0.72], dtype=float)

# Baseline persistence / mean reversion around X0.
LAMBDA = np.diag([0.055, 0.050, 0.050, 0.040, 0.045, 0.050, 0.050, 0.040])

# Directed cross-domain coupling template (receiver=row, source=column).
AC = np.array([
    [0.00, 0.30, 0.16, 0.00, 0.00, 0.00, 0.00, 0.24],
    [0.10, 0.00, 0.12, 0.00, 0.00, 0.00, 0.00, 0.16],
    [0.00, 0.00, 0.00, 0.00, 0.10, 0.28, 0.22, 0.12],
    [0.00, 0.12, 0.18, 0.00, 0.00, 0.00, 0.00, 0.10],
    [0.00, 0.10, 0.12, 0.28, 0.00, 0.00, 0.00, 0.12],
    [0.00, 0.18, 0.12, 0.00, 0.00, 0.00, 0.18, 0.10],
    [0.10, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.18],
    [0.08, 0.00, 0.08, 0.00, 0.00, 0.00, 0.10, 0.00],
], dtype=float)

# Intervention map, columns H,R,HTA,B,V,O.
BU = 0.60 * np.array([
    [0.34, 0.02, 0.02, 0.12, 0.02, 0.08],
    [0.08, 0.04, 0.05, 0.10, 0.04, 0.04],
    [0.06, 0.10, 0.04, 0.24, 0.08, 0.04],
    [0.02, 0.30, 0.10, 0.06, 0.02, 0.02],
    [0.04, 0.08, 0.30, 0.14, 0.02, 0.02],
    [0.02, 0.02, 0.02, 0.04, 0.34, 0.02],
    [0.04, 0.02, 0.02, 0.04, 0.30, 0.10],
    [0.10, 0.02, 0.02, 0.08, 0.04, 0.34],
], dtype=float)

F = np.diag([0.30, 0.26, 0.24, 0.20, 0.22, 0.20, 0.20, 0.28])
DELTA = 0.004
THETA = np.array([0.30, 0.25, 0.35, 0.55, 0.45, 0.25, 0.25, 0.55], dtype=float)
KAPPA = 4.0

# Local autonomous stability correction found by audit.
COUPLING_SCALE = 0.068

# Actor-specific state weights. These operationalise different local emphases.
Q_DIAGS = np.array([
    [5.0, 0.15, 0.8, 0.10, 0.10, 0.05, 0.05, 1.2],  # health services
    [0.15, 0.15, 0.8, 5.0, 0.8, 0.05, 0.05, 0.2],   # regulator
    [0.2, 0.4, 0.3, 1.0, 5.0, 0.05, 0.05, 0.2],     # HTA / access
    [0.8, 0.3, 4.0, 0.5, 1.2, 0.1, 0.1, 0.5],       # budget / policy
    [0.1, 0.2, 0.5, 0.1, 0.1, 4.0, 4.0, 0.2],       # visibility / participation
    [0.8, 0.2, 0.5, 0.2, 0.2, 0.2, 0.8, 5.0],       # organisation / workforce
], dtype=float)
R = np.full(N_U, 14.0)

U_MAX = 0.08
SHARED_BUDGET = 0.18
A_MIN = 0.40
PHI_MAX = 0.12
RHO_A = 400.0
RHO_PHI = 400.0
U_STATUS_QUO = np.array([0.010, 0.006, 0.006, 0.008, 0.006, 0.008], dtype=float)

NOMINAL = ModelParams(
    x0=X0.copy(), xstar=XSTAR.copy(), Lambda=LAMBDA.copy(), Ac=AC.copy(),
    Bu=BU.copy(), F=F.copy(), delta=DELTA, theta=THETA.copy(), kappa=KAPPA,
    coupling_scale=COUPLING_SCALE, q_diags=Q_DIAGS.copy(), r=R.copy(),
    u_max=U_MAX, shared_budget=SHARED_BUDGET, a_min=A_MIN, phi_max=PHI_MAX,
    rho_a=RHO_A, rho_phi=RHO_PHI, status_quo_control=U_STATUS_QUO.copy(),
)
DEFAULT_SOLVER = SolverSettings()


def activation(x: np.ndarray, p: ModelParams = NOMINAL) -> np.ndarray:
    """Positive threshold activation s_i(x)=[tanh(kappa*(x_i-theta_i))]_+."""
    return np.maximum(0.0, np.tanh(p.kappa * (x - p.theta)))


def activation_derivative(x: np.ndarray, p: ModelParams = NOMINAL) -> np.ndarray:
    z = p.kappa * (x - p.theta)
    out = np.zeros_like(z, dtype=float)
    mask = z > 0.0
    out[mask] = p.kappa / np.cosh(z[mask])**2
    return out


def coupling_gain(x_o: float, p: ModelParams = NOMINAL) -> float:
    return p.coupling_scale * (0.80 + 0.40 * float(x_o))


def k_matrix(x_o: float, p: ModelParams = NOMINAL) -> np.ndarray:
    return coupling_gain(x_o, p) * p.Ac


def raw_state_step(
    x: np.ndarray,
    u: np.ndarray,
    p: ModelParams = NOMINAL,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Unprojected state update.

    The centred nonlinear term makes x0 an exact equilibrium when u=w=0.
    """
    if w is None:
        w = np.zeros(N_X)
    z = x - p.x0
    dz = (
        -p.Lambda @ z
        + k_matrix(x[7], p) @ z
        + p.Bu @ u
        + p.delta * p.F @ (activation(x, p) - activation(p.x0, p))
        + w
    )
    return x + dz


def state_step(
    x: np.ndarray,
    u: np.ndarray,
    p: ModelParams = NOMINAL,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.clip(raw_state_step(x, u, p, w), 0.0, 1.0)


def jacobian_autonomous(x: np.ndarray, p: ModelParams = NOMINAL) -> np.ndarray:
    """Jacobian of the unprojected autonomous map x -> f(x,0,0)."""
    g = coupling_gain(x[7], p)
    J = np.eye(N_X) - p.Lambda + g * p.Ac
    J += p.delta * p.F @ np.diag(activation_derivative(x, p))
    # derivative of g(x_O) A_c (x-x0) with respect to x_O
    dg = p.coupling_scale * 0.40
    J[:, 7] += dg * (p.Ac @ (x - p.x0))
    return J


def spectral_radius(x: np.ndarray, p: ModelParams = NOMINAL) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(jacobian_autonomous(x, p)))))


def access_scores(x: np.ndarray) -> np.ndarray:
    """Three illustrative normalised access scores.

    The low-visibility score is projected to [0,1] because its visibility penalty
    otherwise permits values down to -0.1 over the normalised state cube.
    """
    m, s, pp, l, e, v, c, o = x
    a_h = 0.42*m + 0.18*l + 0.18*e + 0.10*c + 0.12*o
    a_l_raw = 0.38*m + 0.16*l + 0.16*e + 0.16*c + 0.14*o - 0.10*(1.0-v)
    a_l = float(np.clip(a_l_raw, 0.0, 1.0))
    a_p = 0.40*m + 0.16*s + 0.20*l + 0.10*c + 0.14*o
    return np.array([a_h, a_l, a_p], dtype=float)


def equity_dispersion(x: np.ndarray) -> float:
    a = access_scores(x)
    return float(np.max(a) - np.min(a))


def model_derived_diagnostic_delay(x: np.ndarray) -> float:
    """Reciprocal proxy used only for interpretation of the medical state."""
    return float(1.0 / max(float(x[0]), 1.0e-9))


def rollout(
    x: np.ndarray,
    U: np.ndarray,
    p: ModelParams = NOMINAL,
    horizon: Optional[int] = None,
) -> np.ndarray:
    if horizon is None:
        horizon = U.shape[1]
    xs = []
    xx = x.copy()
    for k in range(horizon):
        xx = state_step(xx, U[:, k], p)
        xs.append(xx.copy())
    return np.asarray(xs)


def local_objective(
    ui: np.ndarray,
    x: np.ndarray,
    U: np.ndarray,
    actor: int,
    equity: bool,
    p: ModelParams = NOMINAL,
) -> float:
    """Actor-local finite-horizon objective.

    The predicted states are x_{t+k+1|t}, k=0,...,Np-1. Other actors' forecast
    sequences remain fixed. Equity slacks are analytically eliminated:
        xi^a=[a_min-a]_+, xi^phi=[phi-phi_max]_+.
    """
    UU = U.copy()
    UU[actor, :] = ui
    xs = rollout(x, UU, p)
    Qi = np.diag(p.q_diags[actor])
    val = 0.0
    for k, xx in enumerate(xs):
        err = xx - p.xstar
        val += float(err @ Qi @ err + p.r[actor] * ui[k]**2)
        if equity:
            a = access_scores(xx)
            xi_a = np.maximum(0.0, p.a_min - a)
            xi_phi = max(0.0, equity_dispersion(xx) - p.phi_max)
            val += p.rho_a * float(xi_a @ xi_a) + p.rho_phi * xi_phi**2
    return float(val)


def solve_actor(
    x: np.ndarray,
    U: np.ndarray,
    actor: int,
    equity: bool,
    p: ModelParams = NOMINAL,
    settings: SolverSettings = DEFAULT_SOLVER,
):
    """Solve one actor's Np-dimensional local nonlinear MPC problem."""
    other = U.sum(axis=0) - U[actor, :]
    residual_budget = np.maximum(0.0, p.shared_budget - other)
    upper = np.minimum(p.u_max, residual_budget)
    initial = np.minimum(np.maximum(U[actor, :], 0.0), upper)
    bounds = [(0.0, float(ub)) for ub in upper]
    res = minimize(
        local_objective,
        initial,
        args=(x, U, actor, equity, p),
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": settings.slsqp_maxiter,
            "ftol": settings.slsqp_ftol,
            "disp": False,
        },
    )
    return np.asarray(res.x, dtype=float), res


def coordinate_dmpc(
    x: np.ndarray,
    U0: np.ndarray,
    equity: bool,
    p: ModelParams = NOMINAL,
    settings: SolverSettings = DEFAULT_SOLVER,
    order: Optional[Sequence[int]] = None,
):
    """Sequential forecast exchange / iterative best response.

    Each local optimisation owns one actor's sequence only. No central optimiser
    ever controls the complete 6 x Np forecast matrix.
    """
    if order is None:
        order = tuple(range(N_U))
    U = U0.copy()
    failures = 0
    converged = False
    diff = np.inf
    for sweep in range(settings.max_sweeps):
        old = U.copy()
        for actor in order:
            ui, res = solve_actor(x, U, actor, equity, p, settings)
            failures += int(not res.success)
            U[actor, :] = ui
        diff = float(np.max(np.abs(U - old)))
        if diff <= settings.coordination_tol:
            converged = True
            return U, sweep + 1, diff, failures, converged
    return U, settings.max_sweeps, diff, failures, converged


def best_response_residual(
    x: np.ndarray,
    U: np.ndarray,
    equity: bool,
    p: ModelParams = NOMINAL,
    settings: SolverSettings = DEFAULT_SOLVER,
):
    """One frozen-forecast BR pass used only as a convergence diagnostic."""
    max_u_residual = 0.0
    max_objective_gain = 0.0
    failures = 0
    for actor in range(N_U):
        ui, res = solve_actor(x, U, actor, equity, p, settings)
        failures += int(not res.success)
        max_u_residual = max(max_u_residual, float(np.max(np.abs(ui - U[actor, :]))))
        j0 = local_objective(U[actor, :], x, U, actor, equity, p)
        j1 = local_objective(ui, x, U, actor, equity, p)
        max_objective_gain = max(max_objective_gain, float(j0 - j1))
    return max_u_residual, max_objective_gain, failures


def _assert_control_feasibility(u: np.ndarray, p: ModelParams, atol: float = 2e-8):
    assert np.all(u >= -atol), f"negative control: {u}"
    assert np.all(u <= p.u_max + atol), f"individual control bound violated: {u}"
    assert float(np.sum(u)) <= p.shared_budget + atol, f"shared budget violated: {u.sum()}"


def simulate(
    scenario: str,
    p: ModelParams = NOMINAL,
    settings: SolverSettings = DEFAULT_SOLVER,
    order: Optional[Sequence[int]] = None,
    disturbances: Optional[np.ndarray] = None,
    diagnostics: bool = True,
    assert_nominal_feasibility: bool = True,
):
    """Run status_quo, dmpc, or dmpc_equity in receding-horizon form.

    `disturbances[t]` is an realised process disturbance. It is not supplied to
    the finite-horizon prediction model; after each realised state, the next
    DMPC problem is re-solved from the updated shared state estimate x_hat=x.
    """
    if scenario not in {"status_quo", "dmpc", "dmpc_equity"}:
        raise ValueError(f"unknown scenario {scenario!r}")
    if disturbances is None:
        disturbances = np.zeros((settings.periods, N_X))
    disturbances = np.asarray(disturbances, dtype=float)
    if disturbances.shape != (settings.periods, N_X):
        raise ValueError("disturbances must have shape (periods, 8)")

    x = p.x0.copy()
    states = [x.copy()]
    controls = []
    coord = []

    if scenario == "status_quo":
        for t in range(settings.periods):
            u = p.status_quo_control.copy()
            _assert_control_feasibility(u, p)
            controls.append(u)
            x = state_step(x, u, p, disturbances[t])
            if assert_nominal_feasibility:
                assert np.all((x >= -1e-12) & (x <= 1.0 + 1e-12))
            states.append(x.copy())
        return np.asarray(states), np.asarray(controls), coord

    equity = scenario == "dmpc_equity"
    Np = settings.horizon
    U = np.tile(p.status_quo_control[:, None], (1, Np))
    for t in range(settings.periods):
        if t > 0:
            U = np.column_stack([U[:, 1:], U[:, -1]])
        U, sweeps, diff, failures, converged = coordinate_dmpc(
            x, U, equity, p, settings, order
        )
        u = U[:, 0].copy()
        _assert_control_feasibility(u, p)
        br_res, br_gain, br_failures = (np.nan, np.nan, 0)
        if diagnostics:
            br_res, br_gain, br_failures = best_response_residual(
                x, U, equity, p, settings
            )
        raw_next = raw_state_step(x, u, p, disturbances[t])
        controls.append(u)
        coord.append({
            "period": t,
            "sweeps": sweeps,
            "coordination_change_inf": diff,
            "coordination_converged": bool(converged),
            "local_solver_failures": int(failures),
            "best_response_residual_inf": float(br_res),
            "best_response_objective_gain": float(br_gain),
            "best_response_solver_failures": int(br_failures),
            "total_control": float(np.sum(u)),
            "max_individual_control": float(np.max(u)),
            "raw_next_state_min": float(np.min(raw_next)),
            "raw_next_state_max": float(np.max(raw_next)),
        })
        x = np.clip(raw_next, 0.0, 1.0)
        states.append(x.copy())

    return np.asarray(states), np.asarray(controls), coord


def summarise(scenario: str, states: np.ndarray, controls: np.ndarray, coord: Sequence[dict], p: ModelParams = NOMINAL):
    final = states[-1]
    a = access_scores(final)
    min_access_series = np.array([access_scores(xx).min() for xx in states])
    crossing = np.flatnonzero(min_access_series >= p.a_min)
    sr = np.array([spectral_radius(xx, p) for xx in states])
    return {
        "scenario": scenario,
        "final_x_M": float(final[0]),
        "final_x_O": float(final[7]),
        "model_derived_diagnostic_delay_years": model_derived_diagnostic_delay(final),
        "final_phi": float(np.ptp(a)),
        "final_min_access": float(np.min(a)),
        "mean_total_control_per_period": float(np.mean(np.sum(controls, axis=1))),
        "max_total_control": float(np.max(np.sum(controls, axis=1))),
        "first_period_meeting_access_floor": int(crossing[0]) if len(crossing) else None,
        "max_autonomous_jacobian_spectral_radius_along_state_path": float(np.max(sr)),
        "mean_coordination_sweeps": float(np.mean([c["sweeps"] for c in coord])) if coord else None,
        "max_coordination_sweeps": int(max(c["sweeps"] for c in coord)) if coord else None,
        "max_coordination_change_inf": float(max(c["coordination_change_inf"] for c in coord)) if coord else None,
        "max_best_response_residual_inf": float(max(c["best_response_residual_inf"] for c in coord)) if coord else None,
        "max_best_response_objective_gain": float(max(c["best_response_objective_gain"] for c in coord)) if coord else None,
        "coordination_nonconverged_periods": int(sum(not c["coordination_converged"] for c in coord)) if coord else 0,
        "total_local_solver_failures": int(sum(c["local_solver_failures"] + c["best_response_solver_failures"] for c in coord)) if coord else 0,
        "projection_activated_nominally": bool(any(c["raw_next_state_min"] < 0.0 or c["raw_next_state_max"] > 1.0 for c in coord)) if coord else False,
    }


def perturbed_params(rng: np.random.Generator, base: ModelParams = NOMINAL, frac: float = 0.10) -> ModelParams:
    """Moderately perturb nonzero coupling/input coefficients and nonlinear intensity.

    Each nonzero A_c and B_u coefficient receives an independent U(1-frac,1+frac)
    multiplier. Delta receives one independent multiplier from the same range.
    Signs and sparsity are preserved.
    """
    Ac = base.Ac.copy()
    mask = Ac != 0.0
    Ac[mask] *= rng.uniform(1.0-frac, 1.0+frac, size=int(mask.sum()))
    Bu = base.Bu.copy()
    maskb = Bu != 0.0
    Bu[maskb] *= rng.uniform(1.0-frac, 1.0+frac, size=int(maskb.sum()))
    delta = float(base.delta * rng.uniform(1.0-frac, 1.0+frac))
    return replace(base, Ac=Ac, Bu=Bu, delta=delta)


def disturbance_sequence(rng: np.random.Generator, periods: int, sigma: float = 0.0015) -> np.ndarray:
    """Small zero-mean realised process perturbations, clipped at +/-3 sigma."""
    w = rng.normal(0.0, sigma, size=(periods, N_X))
    return np.clip(w, -3.0*sigma, 3.0*sigma)
