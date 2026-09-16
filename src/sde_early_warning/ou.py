"""Linear OU simulator and transparent local estimators.

This module contains the light-weight reference implementation used by both
benchmark scripts.  The observed process may be irregular, noisy and contain
missing values; missing observations are never globally interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class ExperimentConfig:
    horizon: float = 100.0
    transition_time: float = 80.0
    latent_dt: float = 0.05
    kappa_initial: float = 0.80
    kappa_before_transition: float = 0.10
    process_sigma: float = 0.20
    observation_stride: int = 4
    measurement_noise_sd: float = 0.03
    missing_fraction: float = 0.0
    window_width: float = 16.0


BASE_CONFIG = ExperimentConfig()


def smoothstep(progress: np.ndarray) -> np.ndarray:
    """Smooth monotone schedule from zero to one."""
    progress = np.clip(np.asarray(progress, dtype=float), 0.0, 1.0)
    return progress**2 * (3.0 - 2.0 * progress)


def scenario_parameters(time, scenario: str, cfg: ExperimentConfig):
    """Return the true recovery rate and process diffusion."""
    progress = smoothstep(np.asarray(time) / cfg.transition_time)
    declining_kappa = cfg.kappa_initial - (
        cfg.kappa_initial - cfg.kappa_before_transition
    ) * progress
    if scenario == "stationary":
        kappa = np.full_like(progress, cfg.kappa_initial)
        sigma = np.full_like(progress, cfg.process_sigma)
    elif scenario == "stability_loss":
        kappa = declining_kappa
        sigma = np.full_like(progress, cfg.process_sigma)
    elif scenario == "noise_growth":
        kappa = np.full_like(progress, cfg.kappa_initial)
        sigma = cfg.process_sigma * np.sqrt(cfg.kappa_initial / declining_kappa)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return kappa, sigma


def simulate_latent_process(scenario, cfg: ExperimentConfig, random_state):
    """Simulate a slowly varying OU process with exact local transitions."""
    time = np.arange(0.0, cfg.horizon + cfg.latent_dt / 2.0, cfg.latent_dt)
    kappa, sigma = scenario_parameters(time, scenario, cfg)
    state = np.empty_like(time)
    state[0] = random_state.normal(scale=sigma[0] / np.sqrt(2.0 * kappa[0]))
    for index in range(1, len(time)):
        phi = np.exp(-kappa[index - 1] * cfg.latent_dt)
        innovation_variance = (
            sigma[index - 1] ** 2
            * (1.0 - phi**2)
            / (2.0 * kappa[index - 1])
        )
        state[index] = phi * state[index - 1] + random_state.normal(
            scale=np.sqrt(innovation_variance)
        )
    return pd.DataFrame(
        {"time": time, "state": state, "kappa_true": kappa, "sigma_true": sigma}
    )


def observe_process(latent, cfg: ExperimentConfig, random_state):
    """Subsample the latent path and add measurement noise and MCAR gaps."""
    observed = latent.iloc[:: cfg.observation_stride].copy().reset_index(drop=True)
    observed["value"] = observed["state"] + random_state.normal(
        scale=cfg.measurement_noise_sd, size=len(observed)
    )
    if cfg.missing_fraction > 0:
        missing = random_state.random(len(observed)) < cfg.missing_fraction
        missing[[0, -1]] = False
        observed.loc[missing, "value"] = np.nan
    return observed


def simulate_observations(scenario, cfg: ExperimentConfig, random_state):
    return observe_process(
        simulate_latent_process(scenario, cfg, random_state), cfg, random_state
    )


def estimate_local_dynamics(
    time,
    values,
    window_end: float,
    window_width: float,
    measurement_noise_sd: float,
):
    """Estimate classic EWS and moment-based OU parameters in one window."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    selected = (time > window_end - window_width) & (time <= window_end)
    local_time = time[selected]
    local_values = values[selected]
    finite = np.isfinite(local_values)
    result = {
        "window_end": window_end,
        "n_observed": int(finite.sum()),
        "variance": np.nan,
        "ar1": np.nan,
        "skewness": np.nan,
        "recovery_rate": np.nan,
        "coefficient_variation": np.nan,
        "kappa_hat": np.nan,
        "sigma_hat": np.nan,
    }
    if finite.sum() < 12 or len(local_time) < 3:
        return result

    center = np.nanmean(local_values)
    centered = local_values - center
    observed_variance = np.nanvar(local_values, ddof=1)
    observed_sd = np.sqrt(observed_variance)
    adjacent = finite[:-1] & finite[1:]
    if adjacent.sum() < 8 or observed_variance <= 0:
        return result

    lag_covariance = np.mean(centered[:-1][adjacent] * centered[1:][adjacent])
    dt = float(np.median(np.diff(local_time)))
    latent_variance = observed_variance - measurement_noise_sd**2
    ar1 = lag_covariance / observed_variance
    result.update(
        variance=observed_variance,
        ar1=ar1,
        skewness=(
            np.nanmean(centered**3) / observed_sd**3 if observed_sd > 0 else np.nan
        ),
        recovery_rate=(np.log(ar1) / dt if ar1 > 0 else np.nan),
        coefficient_variation=(
            observed_sd / abs(center)
            if observed_sd > 0 and abs(center) > 0.1 * observed_sd
            else np.nan
        ),
    )
    if latent_variance <= 0:
        return result
    phi = lag_covariance / latent_variance
    if not 0.0 < phi < 1.0:
        return result
    kappa = -np.log(phi) / dt
    result.update(kappa_hat=kappa, sigma_hat=np.sqrt(2.0 * kappa * latent_variance))
    return result


def rolling_estimates(observed, cfg: ExperimentConfig, window_ends: Iterable[float]):
    rows = [
        estimate_local_dynamics(
            observed["time"].to_numpy(),
            observed["value"].to_numpy(),
            window_end,
            cfg.window_width,
            cfg.measurement_noise_sd,
        )
        for window_end in window_ends
    ]
    return pd.DataFrame(rows)


KALMAN_BOUNDS = np.log(
    [
        (0.02, 3.0),
        (0.01, 2.0),
        (0.001, 1.0),
    ]
)


def kalman_ou_negative_log_likelihood(log_parameters, time, values):
    """Exact OU Kalman likelihood for irregular steps and missing values."""
    kappa, process_sigma, measurement_noise = np.exp(log_parameters)
    centered = values - np.nanmean(values)
    stationary_variance = process_sigma**2 / (2.0 * kappa)
    filtered_mean = 0.0
    filtered_variance = stationary_variance
    negative_log_likelihood = 0.0
    n_observed = 0
    for index in range(len(time)):
        if index > 0:
            dt = time[index] - time[index - 1]
            phi = np.exp(-kappa * dt)
            innovation_variance = stationary_variance * (1.0 - phi**2)
            filtered_mean = phi * filtered_mean
            filtered_variance = (
                phi**2 * filtered_variance + innovation_variance
            )
        if np.isfinite(centered[index]):
            prediction_variance = filtered_variance + measurement_noise**2
            residual = centered[index] - filtered_mean
            negative_log_likelihood += 0.5 * (
                np.log(2.0 * np.pi * prediction_variance)
                + residual**2 / prediction_variance
            )
            kalman_gain = filtered_variance / prediction_variance
            filtered_mean += kalman_gain * residual
            filtered_variance = max(
                (1.0 - kalman_gain) * filtered_variance, 1e-12
            )
            n_observed += 1
    return negative_log_likelihood if n_observed >= 8 else 1e12


def fit_local_ou_kalman(time, values):
    """Jointly estimate recovery, process diffusion and observation noise."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    observed = values[finite]
    empty = {
        "kappa_hat": np.nan,
        "sigma_hat": np.nan,
        "measurement_noise_hat": np.nan,
        "converged": False,
        "n_observed": int(finite.sum()),
    }
    if finite.sum() < 8:
        return empty

    dt = float(np.median(np.diff(time)))
    observed_variance = float(np.var(observed, ddof=1))
    differences = np.diff(values)
    differences = differences[np.isfinite(differences)]
    tau_initial = max(
        float(np.std(differences) / 4.0) if len(differences) else 0.03, 0.005
    )
    latent_variance = max(observed_variance - tau_initial**2, 1e-4)
    centered = values - np.nanmean(values)
    adjacent = finite[:-1] & finite[1:]
    lag_covariance = (
        float(np.mean(centered[:-1][adjacent] * centered[1:][adjacent]))
        if adjacent.sum() > 4
        else 0.5 * observed_variance
    )
    phi_initial = np.clip(
        lag_covariance / max(observed_variance, 1e-8), 0.10, 0.98
    )
    kappa_initial = np.clip(-np.log(phi_initial) / dt, 0.03, 2.5)
    sigma_initial = np.clip(
        np.sqrt(2.0 * kappa_initial * latent_variance), 0.02, 1.5
    )
    starts = [
        np.log([kappa_initial, sigma_initial, tau_initial]),
        np.log(
            [
                0.5,
                np.sqrt(max(observed_variance, 1e-4)),
                max(0.03, tau_initial),
            ]
        ),
    ]
    candidates = [
        minimize(
            kalman_ou_negative_log_likelihood,
            start,
            args=(time, values),
            method="L-BFGS-B",
            bounds=KALMAN_BOUNDS,
            options={"maxiter": 120},
        )
        for start in starts
    ]
    result = min(candidates, key=lambda candidate: candidate.fun)
    kappa, sigma, tau = np.exp(result.x)
    return {
        "kappa_hat": float(kappa),
        "sigma_hat": float(sigma),
        "measurement_noise_hat": float(tau),
        "converged": bool(result.success and np.isfinite(result.fun)),
        "n_observed": int(finite.sum()),
    }
