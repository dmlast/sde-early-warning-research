"""Generate the cached local SDE discovery benchmark used by the notebook."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sde_early_warning.estimators import fit_discovery_estimators  # noqa: E402
from sde_early_warning.ou import (  # noqa: E402
    BASE_CONFIG,
    estimate_local_dynamics,
    fit_local_ou_kalman,
    simulate_observations,
)

CONDITIONS = {
    "baseline": BASE_CONFIG,
    "combined": replace(
        BASE_CONFIG,
        observation_stride=16,
        measurement_noise_sd=0.12,
        missing_fraction=0.30,
    ),
}


def local_window(observed, cfg, end):
    selected = observed.loc[
        (observed["time"] > end - cfg.window_width) & (observed["time"] <= end)
    ]
    return selected["time"].to_numpy(float), selected["value"].to_numpy(float)


def fit_one_window(time, values, cfg, neural_seed):
    moment = estimate_local_dynamics(
        time, values, float(time[-1]), cfg.window_width, cfg.measurement_noise_sd
    )
    kalman = fit_local_ou_kalman(time, values)
    methods = {
        "Moment OU": {
            "kappa_hat": moment["kappa_hat"],
            "sigma_hat": moment["sigma_hat"],
            "kappa_se": np.nan,
            "active_terms": np.nan,
            "converged": np.isfinite(moment["kappa_hat"]) and np.isfinite(moment["sigma_hat"]),
        },
        "Kalman OU": {
            "kappa_hat": kalman["kappa_hat"],
            "sigma_hat": kalman["sigma_hat"],
            "tau_hat": kalman["measurement_noise_hat"],
            "kappa_se": np.nan,
            "active_terms": np.nan,
            "converged": kalman["converged"],
        },
        **fit_discovery_estimators(time, values, neural_seed=neural_seed),
    }
    return methods


def run_trajectory(task):
    condition, scenario, repetition = task
    cfg = CONDITIONS[condition]
    random_state = np.random.default_rng(
        20260920
        + repetition
        + 1000 * (condition == "combined")
        + 10000 * {"stability_loss": 0, "noise_growth": 1, "stationary": 2}[scenario]
    )
    observed = simulate_observations(scenario, cfg, random_state)
    early_time, early_values = local_window(observed, cfg, 25.0)
    late_time, late_values = local_window(observed, cfg, 78.0)
    early = fit_one_window(early_time, early_values, cfg, 2 * repetition)
    late = fit_one_window(late_time, late_values, cfg, 2 * repetition + 1)
    true_kappa_late = float(
        observed.loc[
            (observed["time"] > 78.0 - cfg.window_width) & (observed["time"] <= 78.0),
            "kappa_true",
        ].mean()
    )
    true_sigma_late = float(
        observed.loc[
            (observed["time"] > 78.0 - cfg.window_width) & (observed["time"] <= 78.0),
            "sigma_true",
        ].mean()
    )
    rows = []
    for method in early:
        rows.append({
            "condition": condition,
            "scenario": scenario,
            "repetition": repetition,
            "method": method,
            "kappa_early": early[method]["kappa_hat"],
            "kappa_late": late[method]["kappa_hat"],
            "sigma_early": early[method]["sigma_hat"],
            "sigma_late": late[method]["sigma_hat"],
            "kappa_decline": early[method]["kappa_hat"] - late[method]["kappa_hat"],
            "sigma_growth": late[method]["sigma_hat"] - early[method]["sigma_hat"],
            "kappa_se_late": late[method].get("kappa_se", np.nan),
            "kappa_lower_late": late[method].get("kappa_lower", np.nan),
            "kappa_upper_late": late[method].get("kappa_upper", np.nan),
            "sigma_lower_late": late[method].get("sigma_lower", np.nan),
            "sigma_upper_late": late[method].get("sigma_upper", np.nan),
            "tau_late": late[method].get("tau_hat", np.nan),
            "tau_true": cfg.measurement_noise_sd,
            "active_terms_late": late[method].get("active_terms", np.nan),
            "pip_x_late": late[method].get("pip_x", np.nan),
            "pip_x2_late": late[method].get("pip_x2", np.nan),
            "pip_x3_late": late[method].get("pip_x3", np.nan),
            "model_entropy_late": late[method].get("model_entropy", np.nan),
            "converged": bool(early[method]["converged"] and late[method]["converged"]),
            "kappa_true_late": true_kappa_late,
            "sigma_true_late": true_sigma_late,
        })
    return rows


def run_sequence(task):
    """Fit every model causally at successive test-time window ends."""
    condition, scenario, repetition = task
    cfg = CONDITIONS[condition]
    random_state = np.random.default_rng(
        20260920
        + repetition
        + 1000 * (condition == "combined")
        + 10000 * {"stability_loss": 0, "noise_growth": 1, "stationary": 2}[scenario]
    )
    observed = simulate_observations(scenario, cfg, random_state)
    early_time, early_values = local_window(observed, cfg, 25.0)
    early = fit_one_window(early_time, early_values, cfg, 100000 + repetition)
    rows = []
    for window_number, window_end in enumerate(np.arange(30.0, 78.1, 8.0)):
        current_time, current_values = local_window(observed, cfg, float(window_end))
        current = fit_one_window(
            current_time,
            current_values,
            cfg,
            200000 + 100 * repetition + window_number,
        )
        for method in early:
            rows.append({
                "condition": condition,
                "scenario": scenario,
                "repetition": repetition,
                "method": method,
                "window_end": window_end,
                "kappa_decline": early[method]["kappa_hat"] - current[method]["kappa_hat"],
                "sigma_growth": current[method]["sigma_hat"] - early[method]["sigma_hat"],
                "converged": bool(early[method]["converged"] and current[method]["converged"]),
            })
    return rows


def main():
    tasks = [
        (condition, scenario, repetition)
        for condition in CONDITIONS
        for scenario in ["stability_loss", "noise_growth", "stationary"]
        for repetition in range(36)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_trajectory, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if completed % 12 == 0:
                print(f"completed {completed}/{len(tasks)} trajectories", flush=True)
    result = pd.DataFrame(rows).sort_values(
        ["condition", "scenario", "repetition", "method"]
    )
    output = ROOT / "data/extended_sde_benchmark.csv"
    result.to_csv(output, index=False)
    print(output, len(result), flush=True)

    sequential_tasks = [
        (condition, scenario, repetition)
        for condition in CONDITIONS
        for scenario in ["stability_loss", "noise_growth"]
        for repetition in range(24, 36)
    ] + [
        (condition, "stationary", repetition)
        for condition in CONDITIONS
        for repetition in range(36)
    ]
    sequential_rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_sequence, task): task for task in sequential_tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            sequential_rows.extend(future.result())
            if completed % 12 == 0:
                print(
                    f"completed sequential {completed}/{len(sequential_tasks)} trajectories",
                    flush=True,
                )
    sequential = pd.DataFrame(sequential_rows).sort_values(
        ["condition", "scenario", "repetition", "method", "window_end"]
    )
    sequential_output = ROOT / "data/extended_sde_sequential.csv"
    sequential.to_csv(sequential_output, index=False)
    print(sequential_output, len(sequential), flush=True)


if __name__ == "__main__":
    main()
