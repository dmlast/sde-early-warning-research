"""Readable local estimators for the stochastic-dynamics benchmarks.

Every estimator receives one short, possibly incomplete observation window.
SINDy uses a cubic candidate library.  Bayesian SINDy performs exact Bayesian
model averaging over all spike-and-slab inclusion patterns.  Two Neural SDE
baselines are provided: a direct Euler likelihood for observed states and a
latent-state variational model with a separate observation-noise distribution.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import torch
from scipy.special import gammaln, logsumexp
from scipy.optimize import brentq
from scipy.stats import invgamma
from torch import nn
from torch.nn import functional as F


POLYNOMIAL_TERMS = ("1", "x", "x²", "x³")


def _increment_data(time: np.ndarray, values: np.ndarray):
    """Return standardized adjacent transitions without interpolating gaps."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    center = float(np.nanmean(values))
    scale = float(np.nanstd(values, ddof=1))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("The local window has no usable variation")

    standardized = (values - center) / scale
    valid = (
        np.isfinite(standardized[:-1])
        & np.isfinite(standardized[1:])
        & np.isfinite(time[:-1])
        & np.isfinite(time[1:])
        & (np.diff(time) > 0)
    )
    x = standardized[:-1][valid]
    x_next = standardized[1:][valid]
    dt = np.diff(time)[valid]
    if len(x) < 8:
        raise ValueError("Fewer than 8 adjacent transitions")
    return x, x_next, dt, center, scale


def polynomial_library(x: np.ndarray) -> np.ndarray:
    """Cubic candidate library Θ(x)=[1,x,x²,x³]."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


def _polynomial_local_stability(coefficients: np.ndarray) -> tuple[float, float]:
    """Return -f'(x*) at the stable real root closest to the window centre."""
    coefficients = np.asarray(coefficients, dtype=float)
    descending = np.trim_zeros(coefficients[::-1], trim="f")
    if len(descending) <= 1:
        return float(-coefficients[1]), 0.0
    roots = np.roots(descending)
    real_roots = roots.real[np.abs(roots.imag) < 1e-6]
    if len(real_roots) == 0:
        return float(-coefficients[1]), 0.0
    derivatives = (
        coefficients[1]
        + 2.0 * coefficients[2] * real_roots
        + 3.0 * coefficients[3] * real_roots**2
    )
    stable = derivatives < 0
    candidates = np.flatnonzero(stable) if stable.any() else np.arange(len(real_roots))
    selected = candidates[np.argmin(np.abs(real_roots[candidates]))]
    return float(-derivatives[selected]), float(real_roots[selected])


def fit_sindy_sde(
    time: np.ndarray,
    values: np.ndarray,
    ridge: float = 1e-3,
    threshold: float = 0.08,
    iterations: int = 8,
) -> dict:
    """Sequential-threshold ridge estimate of drift and residual diffusion."""
    try:
        x, x_next, dt, _, scale = _increment_data(time, values)
        response = (x_next - x) / dt
        library = polynomial_library(x)
        active = np.ones(library.shape[1], dtype=bool)
        active[0] = True
        coefficients = np.zeros(library.shape[1])

        for _ in range(iterations):
            design = library[:, active]
            penalty = ridge * np.eye(design.shape[1])
            estimated = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ response,
            )
            coefficients[:] = 0.0
            coefficients[active] = estimated
            new_active = np.abs(coefficients) >= threshold
            new_active[0] = True
            if np.array_equal(new_active, active):
                break
            active = new_active

        drift = library @ coefficients
        innovations = x_next - x - drift * dt
        diffusion = scale * np.sqrt(np.mean(innovations**2 / dt))
        kappa, equilibrium = _polynomial_local_stability(coefficients)
        return {
            "kappa_hat": kappa,
            "sigma_hat": float(diffusion),
            "kappa_se": np.nan,
            "active_terms": int(active.sum()),
            "equilibrium_hat": equilibrium,
            "equation": " + ".join(
                f"{coefficient:+.3f}{term}"
                for coefficient, term, keep in zip(coefficients, POLYNOMIAL_TERMS, active)
                if keep
            ),
            "objective": float(np.mean((response - drift) ** 2)),
            "converged": bool(np.isfinite(coefficients).all() and np.isfinite(diffusion)),
        }
    except (ValueError, np.linalg.LinAlgError):
        return _empty_result()


def fit_bayesian_sindy_sde(
    time: np.ndarray,
    values: np.ndarray,
    seed: int = 0,
    posterior_draws: int = 800,
    inclusion_probability: float = 0.35,
) -> dict:
    """Exact spike-and-slab Bayesian SINDy for the cubic drift library.

    The intercept is always present and the three nonlinear library terms are
    independently included or excluded.  Because there are only eight models,
    their marginal likelihoods and posterior model probabilities can be
    evaluated exactly.  Conditional coefficient and diffusion posteriors use a
    conjugate Normal--inverse-Gamma prior; returned intervals integrate over
    both parameter and model-selection uncertainty.
    """
    try:
        x, x_next, dt, _, scale = _increment_data(time, values)
        theta = polynomial_library(x)

        # Whiten the Euler increments: Δx/sqrt(Δt) = Θ(x)β sqrt(Δt) + ε,
        # ε ~ N(0, q).  Thus q is the standardized diffusion variance even
        # when observation intervals are irregular.
        response = (x_next - x) / np.sqrt(dt)
        weighted_library = theta * np.sqrt(dt)[:, None]
        a0, b0 = 2.0, 0.05
        models = []
        log_weights = []
        for pattern in product((0, 1), repeat=3):
            included = np.array((1, *pattern), dtype=bool)
            design = weighted_library[:, included]
            prior_covariance = np.eye(design.shape[1]) * 100.0
            prior_precision = np.linalg.inv(prior_covariance)
            posterior_precision = prior_precision + design.T @ design
            posterior_covariance = np.linalg.inv(posterior_precision)
            posterior_mean = posterior_covariance @ design.T @ response
            an = a0 + len(response) / 2.0
            quadratic = response @ response - posterior_mean @ posterior_precision @ posterior_mean
            bn = b0 + 0.5 * max(float(quadratic), 1e-10)
            _, logdet_prior = np.linalg.slogdet(prior_covariance)
            _, logdet_posterior = np.linalg.slogdet(posterior_covariance)
            log_evidence = (
                -0.5 * len(response) * np.log(2.0 * np.pi)
                + 0.5 * (logdet_posterior - logdet_prior)
                + a0 * np.log(b0)
                - an * np.log(bn)
                + gammaln(an)
                - gammaln(a0)
            )
            included_count = int(sum(pattern))
            log_prior_model = (
                included_count * np.log(inclusion_probability)
                + (3 - included_count) * np.log(1.0 - inclusion_probability)
            )
            models.append({
                "included": included,
                "mean": posterior_mean,
                "covariance": posterior_covariance,
                "an": an,
                "bn": bn,
            })
            log_weights.append(log_evidence + log_prior_model)

        probabilities = np.exp(np.asarray(log_weights) - logsumexp(log_weights))
        rng = np.random.default_rng(seed)
        selected_models = rng.choice(len(models), size=posterior_draws, p=probabilities)
        coefficient_draws = np.zeros((posterior_draws, 4), dtype=float)
        diffusion_draws = np.empty(posterior_draws, dtype=float)
        for model_index, model in enumerate(models):
            locations = np.flatnonzero(selected_models == model_index)
            if len(locations) == 0:
                continue
            variance_draws = invgamma.rvs(
                model["an"],
                scale=model["bn"],
                size=len(locations),
                random_state=rng,
            )
            standard_normal = rng.normal(size=(len(locations), len(model["mean"])))
            chol = np.linalg.cholesky(model["covariance"])
            conditional_draws = (
                model["mean"][None, :]
                + np.sqrt(variance_draws)[:, None] * (standard_normal @ chol.T)
            )
            coefficient_draws[np.ix_(locations, model["included"])] = conditional_draws
            diffusion_draws[locations] = scale * np.sqrt(variance_draws)

        inclusion_probabilities = np.array([
            sum(probability for probability, model in zip(probabilities, models) if model["included"][j])
            for j in range(4)
        ])
        coefficient_mean = np.average(coefficient_draws, axis=0)
        stability = np.array([
            _polynomial_local_stability(draw) for draw in coefficient_draws
        ])
        kappa_draws = stability[:, 0]
        active = inclusion_probabilities >= 0.5
        active[0] = True
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-15)))
        return {
            "kappa_hat": float(np.mean(kappa_draws)),
            "sigma_hat": float(np.mean(diffusion_draws)),
            "kappa_se": float(np.std(kappa_draws, ddof=1)),
            "kappa_lower": float(np.quantile(kappa_draws, 0.025)),
            "kappa_upper": float(np.quantile(kappa_draws, 0.975)),
            "sigma_lower": float(np.quantile(diffusion_draws, 0.025)),
            "sigma_upper": float(np.quantile(diffusion_draws, 0.975)),
            "active_terms": int(active.sum()),
            "equilibrium_hat": float(np.mean(stability[:, 1])),
            "pip_x": float(inclusion_probabilities[1]),
            "pip_x2": float(inclusion_probabilities[2]),
            "pip_x3": float(inclusion_probabilities[3]),
            "model_entropy": entropy,
            "equation": " + ".join(
                f"{coefficient:+.3f}{term}"
                for coefficient, term, keep in zip(coefficient_mean, POLYNOMIAL_TERMS, active)
                if keep
            ),
            "objective": float(-logsumexp(log_weights)),
            "converged": bool(
                np.isfinite(coefficient_draws).all() and np.isfinite(diffusion_draws).all()
            ),
        }
    except (ValueError, np.linalg.LinAlgError):
        return _empty_result()


class LocalNeuralSDE(nn.Module):
    """Small neural drift/diffusion model for one-dimensional local windows."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.drift_linear = nn.Linear(1, 1)
        self.drift_residual = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.log_diffusion = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def drift(self, x):
        return self.drift_linear(x) + 0.1 * self.drift_residual(x)

    def diffusion(self, x):
        return F.softplus(self.log_diffusion(x)) + 1e-4

    def forward(self, x, dt):
        mean = x + self.drift(x) * dt
        variance = self.diffusion(x).square() * dt + 1e-6
        return mean, variance


def _neural_local_stability(drift_model) -> tuple[float, float]:
    """Find the stable neural-drift root nearest zero and evaluate -f' there."""
    grid = np.linspace(-3.0, 3.0, 121)

    def scalar_drift(value):
        with torch.no_grad():
            point = torch.tensor([[value]], dtype=torch.float32)
            return float(drift_model(point).item())

    values = np.array([scalar_drift(value) for value in grid])
    roots = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left == 0:
            roots.append(left)
        elif np.sign(f_left) != np.sign(f_right):
            roots.append(brentq(scalar_drift, left, right))
    if not roots:
        roots = [0.0]
    candidates = []
    for root in roots:
        point = torch.tensor([[root]], dtype=torch.float32, requires_grad=True)
        derivative = torch.autograd.grad(drift_model(point).sum(), point)[0].item()
        candidates.append((root, derivative))
    stable = [candidate for candidate in candidates if candidate[1] < 0]
    root, derivative = min(stable or candidates, key=lambda item: abs(item[0]))
    return float(-derivative), float(root)


def fit_neural_sde(
    time: np.ndarray,
    values: np.ndarray,
    seed: int = 0,
    steps: int = 220,
    learning_rate: float = 1e-2,
) -> dict:
    """Fit an observed-state Neural SDE by Gaussian Euler likelihood."""
    try:
        x, x_next, dt, _, scale = _increment_data(time, values)
    except ValueError:
        return _empty_result()

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = LocalNeuralSDE()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    next_tensor = torch.tensor(x_next, dtype=torch.float32).reshape(-1, 1)
    dt_tensor = torch.tensor(dt, dtype=torch.float32).reshape(-1, 1)
    final_loss = np.nan

    for _ in range(steps):
        optimizer.zero_grad()
        mean, variance = model(x_tensor, dt_tensor)
        nll = 0.5 * (torch.log(variance) + (next_tensor - mean).square() / variance)
        equilibrium = torch.zeros((1, 1), dtype=torch.float32)
        regularizer = 0.01 * model.drift(equilibrium).square().mean()
        loss = nll.mean() + regularizer
        if not torch.isfinite(loss):
            return _empty_result()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        final_loss = float(loss.detach())

    kappa, equilibrium = _neural_local_stability(model.drift)
    with torch.no_grad():
        diffusion = float(model.diffusion(torch.zeros((1, 1))).item() * scale)
    return {
        "kappa_hat": kappa,
        "sigma_hat": diffusion,
        "kappa_se": np.nan,
        "active_terms": np.nan,
        "equilibrium_hat": equilibrium,
        "equation": "fθ(x), gθ(x)",
        "objective": final_loss,
        "converged": bool(np.isfinite(kappa) and np.isfinite(diffusion)),
    }


class LatentNeuralSDE(nn.Module):
    """Neural drift/diffusion plus an explicit Gaussian observation model."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.dynamics = LocalNeuralSDE(hidden=hidden)
        self.raw_observation_sd = nn.Parameter(torch.tensor(-2.0))

    def observation_sd(self):
        return F.softplus(self.raw_observation_sd) + 1e-4


def fit_latent_neural_sde(
    time: np.ndarray,
    values: np.ndarray,
    seed: int = 0,
    steps: int = 260,
    learning_rate: float = 8e-3,
) -> dict:
    """Fit a variational latent Neural SDE with Gaussian measurements.

    A factorized Gaussian posterior is optimized for every latent state.  The
    ELBO contains the Euler transition density, a separate observation density,
    a weak initial-state prior and the variational entropy.  Missing observations
    remain latent and therefore do not break the transition sequence.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    observed_mask = np.isfinite(values) & np.isfinite(time)
    if observed_mask.sum() < 8 or len(time) < 9 or np.any(np.diff(time) <= 0):
        return _empty_result()
    center = float(np.nanmean(values))
    scale = float(np.nanstd(values, ddof=1))
    if not np.isfinite(scale) or scale < 1e-8:
        return _empty_result()
    standardized = (values - center) / scale
    indices = np.arange(len(values))
    initial_path = np.interp(indices, indices[observed_mask], standardized[observed_mask])

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = LatentNeuralSDE()
    latent_mean = nn.Parameter(torch.tensor(initial_path, dtype=torch.float32).reshape(-1, 1))
    latent_log_sd = nn.Parameter(torch.full((len(values), 1), -2.3, dtype=torch.float32))
    parameters = list(model.parameters()) + [latent_mean, latent_log_sd]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
    observations = torch.tensor(
        np.nan_to_num(standardized, nan=0.0), dtype=torch.float32
    ).reshape(-1, 1)
    mask = torch.tensor(observed_mask, dtype=torch.bool).reshape(-1, 1)
    dt = torch.tensor(np.diff(time), dtype=torch.float32).reshape(-1, 1)
    final_loss = np.nan

    for _ in range(steps):
        optimizer.zero_grad()
        posterior_sd = F.softplus(latent_log_sd) + 1e-3
        latent = latent_mean + posterior_sd * torch.randn_like(latent_mean)
        transition_mean, transition_variance = model.dynamics(latent[:-1], dt)
        transition_nll = 0.5 * (
            torch.log(transition_variance)
            + (latent[1:] - transition_mean).square() / transition_variance
            + np.log(2.0 * np.pi)
        ).sum()
        observation_variance = model.observation_sd().square()
        observation_nll = 0.5 * (
            torch.log(observation_variance)
            + (observations[mask] - latent[mask]).square() / observation_variance
            + np.log(2.0 * np.pi)
        ).sum()
        initial_prior = 0.5 * (latent[0].square() + np.log(2.0 * np.pi)).sum()
        entropy = (
            torch.log(posterior_sd) + 0.5 * (1.0 + np.log(2.0 * np.pi))
        ).sum()
        equilibrium_penalty = 0.01 * model.dynamics.drift(
            torch.zeros((1, 1), dtype=torch.float32)
        ).square().sum()
        loss = (
            transition_nll + observation_nll + initial_prior - entropy
        ) / len(values) + equilibrium_penalty
        if not torch.isfinite(loss):
            return _empty_result()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
        final_loss = float(loss.detach())

    kappa, equilibrium = _neural_local_stability(model.dynamics.drift)
    with torch.no_grad():
        diffusion = float(model.dynamics.diffusion(torch.zeros((1, 1))).item() * scale)
        observation_sd = float(model.observation_sd().item() * scale)
    return {
        "kappa_hat": kappa,
        "sigma_hat": diffusion,
        "tau_hat": observation_sd,
        "kappa_se": np.nan,
        "active_terms": np.nan,
        "equilibrium_hat": equilibrium,
        "equation": "z: dz=fθ(z)dt+gθ(z)dW; y|z~N(z,τ²)",
        "objective": final_loss,
        "converged": bool(
            np.isfinite(kappa) and np.isfinite(diffusion) and np.isfinite(observation_sd)
        ),
    }


def _empty_result() -> dict:
    return {
        "kappa_hat": np.nan,
        "sigma_hat": np.nan,
        "tau_hat": np.nan,
        "kappa_se": np.nan,
        "kappa_lower": np.nan,
        "kappa_upper": np.nan,
        "sigma_lower": np.nan,
        "sigma_upper": np.nan,
        "active_terms": np.nan,
        "equilibrium_hat": np.nan,
        "pip_x": np.nan,
        "pip_x2": np.nan,
        "pip_x3": np.nan,
        "model_entropy": np.nan,
        "equation": "",
        "objective": np.nan,
        "converged": False,
    }


def fit_discovery_estimators(time, values, neural_seed=0) -> dict[str, dict]:
    """Run all non-OU discovery estimators on one local window."""
    return {
        "SINDy-STLSQ": fit_sindy_sde(time, values),
        "Bayesian SINDy (spike-slab)": fit_bayesian_sindy_sde(
            time, values, seed=neural_seed
        ),
        "Neural SDE (Euler)": fit_neural_sde(time, values, seed=neural_seed),
        "Latent Neural SDE (VI)": fit_latent_neural_sde(
            time, values, seed=neural_seed
        ),
    }
