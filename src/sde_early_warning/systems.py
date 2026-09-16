"""Controlled nonlinear SDE systems used by the synthetic benchmark.

The three normal forms are parameterized by the same target local recovery
rate kappa(t).  This keeps the distance to the transition comparable while the
global drift geometry remains genuinely different.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


SYSTEMS = ("saddle_node", "transcritical", "bistable")
SCENARIOS = ("stability_loss", "noise_growth", "stationary")


@dataclass(frozen=True)
class NonlinearConfig:
    horizon: float = 90.0
    transition_time: float = 80.0
    latent_dt: float = 0.01
    kappa_initial: float = 0.80
    kappa_before_transition: float = 0.10
    process_sigma: float = 0.03
    observation_stride: int = 20
    measurement_noise_sd: float = 0.03
    missing_fraction: float = 0.0
    window_width: float = 16.0


BASE_NONLINEAR_CONFIG = NonlinearConfig()
COMBINED_NONLINEAR_CONFIG = replace(
    BASE_NONLINEAR_CONFIG,
    observation_stride=80,
    measurement_noise_sd=0.08,
    missing_fraction=0.30,
)


def smoothstep(progress):
    progress = np.clip(np.asarray(progress, dtype=float), 0.0, 1.0)
    return progress**2 * (3.0 - 2.0 * progress)


def scheduled_kappa(time, cfg: NonlinearConfig):
    progress = smoothstep(np.asarray(time) / cfg.transition_time)
    return cfg.kappa_initial - (
        cfg.kappa_initial - cfg.kappa_before_transition
    ) * progress


def equilibrium_and_control(system: str, kappa):
    """Return the attracting equilibrium and normal-form control parameter."""
    kappa = np.asarray(kappa, dtype=float)
    if system == "saddle_node":
        equilibrium = kappa / 2.0
        control = equilibrium**2
    elif system == "transcritical":
        equilibrium = kappa
        control = kappa
    elif system == "bistable":
        equilibrium = np.sqrt((kappa + 1.0) / 3.0)
        control = equilibrium**3 - equilibrium
    else:
        raise ValueError(f"Unknown nonlinear system: {system}")
    return equilibrium, control


def drift(system: str, state, control):
    state = np.asarray(state)
    if system == "saddle_node":
        return control - state**2
    if system == "transcritical":
        return control * state - state**2
    if system == "bistable":
        return control + state - state**3
    raise ValueError(f"Unknown nonlinear system: {system}")


def state_dependent_diffusion(state, equilibrium, amplitude):
    """Positive diffusion whose value at the moving equilibrium is amplitude."""
    return amplitude * (1.0 + 0.30 * np.tanh(np.asarray(state) - equilibrium))


def scenario_parameters(time, system: str, scenario: str, cfg: NonlinearConfig):
    changing_kappa = scheduled_kappa(time, cfg)
    if scenario == "stability_loss":
        kappa = changing_kappa
        amplitude = np.full_like(kappa, cfg.process_sigma)
    elif scenario == "noise_growth":
        kappa = np.full_like(changing_kappa, cfg.kappa_initial)
        # Match the local-linear stationary variance sigma²/(2*kappa).
        amplitude = cfg.process_sigma * np.sqrt(cfg.kappa_initial / changing_kappa)
    elif scenario == "stationary":
        kappa = np.full_like(changing_kappa, cfg.kappa_initial)
        amplitude = np.full_like(kappa, cfg.process_sigma)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    equilibrium, control = equilibrium_and_control(system, kappa)
    return kappa, amplitude, equilibrium, control


def simulate_latent_process(
    system: str,
    scenario: str,
    cfg: NonlinearConfig,
    random_state: np.random.Generator,
):
    """Euler--Maruyama simulation with an explicit clipping diagnostic."""
    time = np.arange(0.0, cfg.horizon + cfg.latent_dt / 2.0, cfg.latent_dt)
    kappa, amplitude, equilibrium, control = scenario_parameters(
        time, system, scenario, cfg
    )
    state = np.empty_like(time)
    state[0] = equilibrium[0] + random_state.normal(
        scale=amplitude[0] / np.sqrt(2.0 * kappa[0])
    )
    clipped = np.zeros(len(time), dtype=bool)
    bounds = {
        "saddle_node": (-1.5, 1.5),
        "transcritical": (-1.0, 2.0),
        "bistable": (-2.0, 2.0),
    }[system]
    sqrt_dt = np.sqrt(cfg.latent_dt)
    for index in range(len(time) - 1):
        diffusion = state_dependent_diffusion(
            state[index], equilibrium[index], amplitude[index]
        )
        next_state = (
            state[index]
            + drift(system, state[index], control[index]) * cfg.latent_dt
            + diffusion * sqrt_dt * random_state.normal()
        )
        if next_state < bounds[0] or next_state > bounds[1] or not np.isfinite(next_state):
            clipped[index + 1] = True
            next_state = np.clip(np.nan_to_num(next_state), *bounds)
        state[index + 1] = next_state

    realized_diffusion = state_dependent_diffusion(state, equilibrium, amplitude)
    return pd.DataFrame({
        "time": time,
        "state": state,
        "kappa_true": kappa,
        "sigma_true": realized_diffusion,
        "sigma_equilibrium": amplitude,
        "equilibrium": equilibrium,
        "control": control,
        "clipped": clipped,
    })


def observe_process(latent, cfg: NonlinearConfig, random_state):
    observed = latent.iloc[:: cfg.observation_stride].copy().reset_index(drop=True)
    observed["value"] = observed["state"] + random_state.normal(
        scale=cfg.measurement_noise_sd, size=len(observed)
    )
    if cfg.missing_fraction > 0:
        missing = random_state.random(len(observed)) < cfg.missing_fraction
        missing[[0, -1]] = False
        observed.loc[missing, "value"] = np.nan
    return observed


def simulate_observations(system, scenario, cfg, random_state):
    latent = simulate_latent_process(system, scenario, cfg, random_state)
    return observe_process(latent, cfg, random_state)
