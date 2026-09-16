"""Generate cached benchmarks for three nonlinear transition normal forms."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sde_early_warning.systems import (  # noqa: E402
    BASE_NONLINEAR_CONFIG,
    COMBINED_NONLINEAR_CONFIG,
    SCENARIOS,
    SYSTEMS,
    simulate_observations,
)
from sde_early_warning.estimators import fit_discovery_estimators  # noqa: E402
from sde_early_warning.ou import estimate_local_dynamics, fit_local_ou_kalman  # noqa: E402

CONDITIONS = {
    "baseline": BASE_NONLINEAR_CONFIG,
    "combined": COMBINED_NONLINEAR_CONFIG,
}
REPETITIONS = 30
TRAIN_REPETITIONS = 20


def local_window(observed, cfg, end):
    frame = observed.loc[
        (observed["time"] > end - cfg.window_width) & (observed["time"] <= end)
    ]
    return frame["time"].to_numpy(float), frame["value"].to_numpy(float)


def fit_one_window(time, values, cfg, neural_seed):
    moment = estimate_local_dynamics(
        time, values, float(time[-1]), cfg.window_width, cfg.measurement_noise_sd
    )
    kalman = fit_local_ou_kalman(time, values)
    return {
        "Moment OU": {
            "kappa_hat": moment["kappa_hat"],
            "sigma_hat": moment["sigma_hat"],
            "converged": np.isfinite(moment["kappa_hat"]) and np.isfinite(moment["sigma_hat"]),
        },
        "Kalman OU": {
            "kappa_hat": kalman["kappa_hat"],
            "sigma_hat": kalman["sigma_hat"],
            "tau_hat": kalman["measurement_noise_hat"],
            "converged": kalman["converged"],
        },
        **fit_discovery_estimators(time, values, neural_seed=neural_seed),
    }


def seed_for(condition, system, scenario, repetition):
    return (
        20261011
        + repetition
        + 1000 * (condition == "combined")
        + 10000 * SYSTEMS.index(system)
        + 100000 * SCENARIOS.index(scenario)
    )


def run_trajectory(task):
    condition, system, scenario, repetition = task
    cfg = CONDITIONS[condition]
    seed = seed_for(condition, system, scenario, repetition)
    observed = simulate_observations(system, scenario, cfg, np.random.default_rng(seed))
    early_time, early_values = local_window(observed, cfg, 25.0)
    late_time, late_values = local_window(observed, cfg, 78.0)
    early = fit_one_window(early_time, early_values, cfg, 2 * seed)
    late = fit_one_window(late_time, late_values, cfg, 2 * seed + 1)
    late_truth = observed.loc[
        (observed["time"] > 78.0 - cfg.window_width) & (observed["time"] <= 78.0)
    ]
    rows = []
    for method in early:
        rows.append({
            "condition": condition,
            "system": system,
            "scenario": scenario,
            "repetition": repetition,
            "method": method,
            "kappa_early": early[method].get("kappa_hat", np.nan),
            "kappa_late": late[method].get("kappa_hat", np.nan),
            "sigma_early": early[method].get("sigma_hat", np.nan),
            "sigma_late": late[method].get("sigma_hat", np.nan),
            "tau_late": late[method].get("tau_hat", np.nan),
            "kappa_decline": early[method].get("kappa_hat", np.nan)
            - late[method].get("kappa_hat", np.nan),
            "sigma_growth": late[method].get("sigma_hat", np.nan)
            - early[method].get("sigma_hat", np.nan),
            "kappa_lower_late": late[method].get("kappa_lower", np.nan),
            "kappa_upper_late": late[method].get("kappa_upper", np.nan),
            "sigma_lower_late": late[method].get("sigma_lower", np.nan),
            "sigma_upper_late": late[method].get("sigma_upper", np.nan),
            "pip_x_late": late[method].get("pip_x", np.nan),
            "pip_x2_late": late[method].get("pip_x2", np.nan),
            "pip_x3_late": late[method].get("pip_x3", np.nan),
            "active_terms_late": late[method].get("active_terms", np.nan),
            "converged": bool(
                early[method].get("converged", False)
                and late[method].get("converged", False)
            ),
            "kappa_true_late": float(late_truth["kappa_true"].mean()),
            "sigma_true_late": float(late_truth["sigma_true"].mean()),
            "tau_true": cfg.measurement_noise_sd,
            "clip_fraction": float(observed.loc[observed["time"] <= 78.0, "clipped"].mean()),
        })
    return rows


def run_sequence(task):
    condition, system, scenario, repetition = task
    cfg = CONDITIONS[condition]
    seed = seed_for(condition, system, scenario, repetition)
    observed = simulate_observations(system, scenario, cfg, np.random.default_rng(seed))
    early_time, early_values = local_window(observed, cfg, 25.0)
    early = fit_one_window(early_time, early_values, cfg, 3000000 + seed)
    rows = []
    for window_number, window_end in enumerate([38.0, 54.0, 70.0, 78.0]):
        current_time, current_values = local_window(observed, cfg, window_end)
        current = fit_one_window(
            current_time, current_values, cfg, 4000000 + 10 * seed + window_number
        )
        for method in early:
            rows.append({
                "condition": condition,
                "system": system,
                "scenario": scenario,
                "repetition": repetition,
                "method": method,
                "window_end": window_end,
                "kappa_decline": early[method].get("kappa_hat", np.nan)
                - current[method].get("kappa_hat", np.nan),
                "sigma_growth": current[method].get("sigma_hat", np.nan)
                - early[method].get("sigma_hat", np.nan),
                "converged": bool(
                    early[method].get("converged", False)
                    and current[method].get("converged", False)
                ),
            })
    return rows


def collect(tasks, worker, label):
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if completed % 18 == 0:
                print(f"{label}: {completed}/{len(tasks)}", flush=True)
    return rows


def main():
    tasks = [
        (condition, system, scenario, repetition)
        for condition in CONDITIONS
        for system in SYSTEMS
        for scenario in SCENARIOS
        for repetition in range(REPETITIONS)
    ]
    benchmark = pd.DataFrame(collect(tasks, run_trajectory, "main")).sort_values(
        ["condition", "system", "scenario", "repetition", "method"]
    )
    output = ROOT / "data/nonlinear_sde_benchmark.csv"
    benchmark.to_csv(output, index=False)
    print(output, len(benchmark), flush=True)

    sequential_tasks = [
        (condition, system, scenario, repetition)
        for condition in CONDITIONS
        for system in SYSTEMS
        for scenario in ("stability_loss", "noise_growth")
        for repetition in range(TRAIN_REPETITIONS, REPETITIONS)
    ] + [
        (condition, system, "stationary", repetition)
        for condition in CONDITIONS
        for system in SYSTEMS
        for repetition in range(REPETITIONS)
    ]
    sequential = pd.DataFrame(
        collect(sequential_tasks, run_sequence, "sequential")
    ).sort_values(["condition", "system", "scenario", "repetition", "method", "window_end"])
    sequential_output = ROOT / "data/nonlinear_sde_sequential.csv"
    sequential.to_csv(sequential_output, index=False)
    print(sequential_output, len(sequential), flush=True)


if __name__ == "__main__":
    main()
