"""Create deterministic figures and LaTeX tables for the final report."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

METHOD_SHORT = {
    "Moment OU": "Moment OU",
    "Kalman OU": "Kalman OU",
    "SINDy-STLSQ": "SINDy",
    "Bayesian SINDy (spike-slab)": "Bayes SINDy",
    "Neural SDE (Euler)": "Neural SDE (набл.)",
    "Latent Neural SDE (VI)": "Neural SDE (скр.)",
}

DISPLAY_VALUES = {
    "condition": {"baseline": "базовый", "combined": "стрессовый"},
    "system": {
        "bistable": "бистабильная",
        "saddle_node": "седло-узел",
        "transcritical": "транскритическая",
    },
    "target": {"mean": "среднее", "variance": "дисперсия"},
}


def prepare(frame):
    frame = frame.copy()
    frame["usable"] = (
        frame["converged"]
        & frame[["kappa_decline", "sigma_growth"]].notna().all(axis=1)
    )
    return frame


def discrimination(frame, group_keys, split):
    rows, classifiers = [], {}
    for key, block in frame.groupby(group_keys):
        key = key if isinstance(key, tuple) else (key,)
        train = block.query(
            "repetition < @split and scenario in ['stability_loss', 'noise_growth'] and usable"
        )
        test = block.query(
            "repetition >= @split and scenario in ['stability_loss', 'noise_growth'] and usable"
        )
        if train["scenario"].nunique() < 2 or test["scenario"].nunique() < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20260911))
        features = ["kappa_decline", "sigma_growth"]
        model.fit(train[features], train["scenario"].eq("stability_loss"))
        probability = model.predict_proba(test[features])[:, 1]
        classifiers[key] = model
        rows.append({
            **dict(zip(group_keys, key)),
            "AUC": roc_auc_score(test["scenario"].eq("stability_loss"), probability),
            "test_coverage": len(test) / (2 * (block["repetition"].max() + 1 - split)),
        })
    return pd.DataFrame(rows), classifiers


def alarm_metrics(sequence, classifiers, group_keys, split, last_rep):
    scored_parts = []
    for key, block in sequence.groupby(group_keys):
        key = key if isinstance(key, tuple) else (key,)
        if key not in classifiers:
            continue
        usable = block["converged"] & block[["kappa_decline", "sigma_growth"]].notna().all(axis=1)
        scored = block.loc[usable].copy()
        scored["probability"] = classifiers[key].predict_proba(
            scored[["kappa_decline", "sigma_growth"]]
        )[:, 1]
        scored_parts.append(scored)
    scored = pd.concat(scored_parts, ignore_index=True)
    thresholds = {}
    for key, block in scored.query("scenario == 'stationary' and repetition < @split").groupby(group_keys):
        key = key if isinstance(key, tuple) else (key,)
        thresholds[key] = float(np.quantile(block.groupby("repetition")["probability"].max(), .95))
    rows = []
    for key, block in scored.groupby(group_keys):
        key = key if isinstance(key, tuple) else (key,)
        threshold = thresholds[key]
        for scenario, scenario_block in block.groupby("scenario"):
            alarm = []
            for repetition in range(split, last_rep):
                path = scenario_block.query("repetition == @repetition")
                alarm.append(bool((path["probability"] > threshold).any()))
            rows.append({**dict(zip(group_keys, key)), "scenario": scenario, "alarm_rate": np.mean(alarm)})
    return pd.DataFrame(rows)


def parameter_metrics(frame, group_keys, split):
    test = frame.query("repetition >= @split and scenario != 'stationary' and usable").copy()
    test["kappa_error"] = (test["kappa_late"] - test["kappa_true_late"]) ** 2
    test["sigma_error"] = (test["sigma_late"] - test["sigma_true_late"]) ** 2
    test["tau_error"] = (test["tau_late"] - test["tau_true"]) ** 2
    metrics = (
        test.groupby(group_keys)
        .agg(
            kappa_RMSE=("kappa_error", lambda x: np.sqrt(x.mean())),
            sigma_RMSE=("sigma_error", lambda x: np.sqrt(x.mean())),
            tau_RMSE=("tau_error", lambda x: np.sqrt(x.mean()) if x.notna().any() else np.nan),
            n_test=("repetition", "size"),
        ).reset_index()
    )
    coverage = frame.groupby(group_keys)["usable"].mean().rename("fit_coverage").reset_index()
    return metrics.merge(coverage, on=group_keys, how="left"), test


def latex_table(frame, filename, columns, labels, formats=None):
    shown = frame[columns].copy()
    for column, mapping in DISPLAY_VALUES.items():
        if column in shown:
            shown[column] = shown[column].replace(mapping)
    formats = formats or {}
    for column, formatter in formats.items():
        shown[column] = shown[column].map(formatter)
    shown = shown.rename(columns=dict(zip(columns, labels)))
    text = shown.to_latex(index=False, escape=True, na_rep="--", column_format="l" * len(columns))
    (TABLES / filename).write_text(text, encoding="utf-8")


extended = prepare(pd.read_csv(DATA / "extended_sde_benchmark.csv"))
extended_sequence = pd.read_csv(DATA / "extended_sde_sequential.csv")
ou_parameters, ou_test = parameter_metrics(extended, ["condition", "method"], 24)
ou_auc, ou_classifiers = discrimination(extended, ["condition", "method"], 24)
ou_alarm = alarm_metrics(extended_sequence, ou_classifiers, ["condition", "method"], 24, 36)
ou_alarm_table = ou_alarm.pivot_table(
    index=["condition", "method"], columns="scenario", values="alarm_rate"
).reset_index()
ou_alarm_table["method_short"] = ou_alarm_table["method"].map(METHOD_SHORT)
latex_table(
    ou_alarm_table,
    "ou_alarms.tex",
    ["condition", "method_short", "stability_loss", "noise_growth", "stationary"],
    ["Режим", "Метод", "Обнаружение", "Ложн.: шум", "Ложн.: стац."],
    {name: (lambda x: f"{x:.0%}") for name in ["stability_loss", "noise_growth", "stationary"]},
)
ou_summary = ou_parameters.merge(ou_auc, on=["condition", "method"], how="left")
ou_summary["method_short"] = ou_summary["method"].map(METHOD_SHORT)
latex_table(
    ou_summary.sort_values(["condition", "kappa_RMSE"]),
    "ou_summary.tex",
    ["condition", "method_short", "fit_coverage", "kappa_RMSE", "sigma_RMSE", "tau_RMSE", "AUC"],
    ["Режим", "Метод", "Успеш. оц.", "κ RMSE", "σ RMSE", "τ RMSE", "AUC"],
    {
        "fit_coverage": lambda x: f"{x:.0%}", "kappa_RMSE": lambda x: f"{x:.3f}",
        "sigma_RMSE": lambda x: f"{x:.3f}", "tau_RMSE": lambda x: "--" if pd.isna(x) else f"{x:.3f}",
        "AUC": lambda x: f"{x:.3f}",
    },
)

bayes_ou = ou_test.query("method == 'Bayesian SINDy (spike-slab)'").copy()
bayes_ou["kappa_coverage"] = (
    (bayes_ou["kappa_true_late"] >= bayes_ou["kappa_lower_late"])
    & (bayes_ou["kappa_true_late"] <= bayes_ou["kappa_upper_late"])
)
bayes_ou["width"] = bayes_ou["kappa_upper_late"] - bayes_ou["kappa_lower_late"]
bayes_ou_calibration = bayes_ou.groupby("condition").agg(
    kappa_coverage=("kappa_coverage", "mean"),
    width=("width", "median"),
).reset_index()
bayes_ou_calibration["system"] = "OU"

nonlinear = prepare(pd.read_csv(DATA / "nonlinear_sde_benchmark.csv"))
nonlinear_sequence = pd.read_csv(DATA / "nonlinear_sde_sequential.csv")
nl_parameters, nl_test = parameter_metrics(nonlinear, ["condition", "system", "method"], 20)
nl_auc, nl_classifiers = discrimination(nonlinear, ["condition", "system", "method"], 20)
nl_alarm = alarm_metrics(nonlinear_sequence, nl_classifiers, ["condition", "system", "method"], 20, 30)
nl_alarm_average = (
    nl_alarm.groupby(["condition", "method", "scenario"])["alarm_rate"].mean()
    .unstack("scenario").reset_index()
)
nl_alarm_average["method_short"] = nl_alarm_average["method"].map(METHOD_SHORT)
latex_table(
    nl_alarm_average,
    "nonlinear_alarms.tex",
    ["condition", "method_short", "stability_loss", "noise_growth", "stationary"],
    ["Режим", "Метод", "Обнаружение", "Ложн.: шум", "Ложн.: стац."],
    {name: (lambda x: f"{x:.0%}") for name in ["stability_loss", "noise_growth", "stationary"]},
)
nl_summary = nl_parameters.merge(nl_auc, on=["condition", "system", "method"], how="left")
nl_average = (
    nl_summary.groupby(["condition", "method"])
    .agg(
        fit_coverage=("fit_coverage", "mean"), kappa_RMSE=("kappa_RMSE", "mean"),
        sigma_RMSE=("sigma_RMSE", "mean"), tau_RMSE=("tau_RMSE", "mean"), AUC=("AUC", "mean"),
    ).reset_index()
)
nl_average["method_short"] = nl_average["method"].map(METHOD_SHORT)
latex_table(
    nl_average.sort_values(["condition", "AUC"], ascending=[True, False]),
    "nonlinear_summary.tex",
    ["condition", "method_short", "fit_coverage", "kappa_RMSE", "sigma_RMSE", "tau_RMSE", "AUC"],
    ["Режим", "Метод", "Успеш. оц.", "κ RMSE", "σ RMSE", "τ RMSE", "Сред. AUC"],
    {
        "fit_coverage": lambda x: f"{x:.0%}", "kappa_RMSE": lambda x: f"{x:.3f}",
        "sigma_RMSE": lambda x: f"{x:.3f}", "tau_RMSE": lambda x: "--" if pd.isna(x) else f"{x:.3f}",
        "AUC": lambda x: f"{x:.3f}",
    },
)

bayes_nl = nl_test.query("method == 'Bayesian SINDy (spike-slab)'").copy()
bayes_nl["kappa_coverage"] = (
    (bayes_nl["kappa_true_late"] >= bayes_nl["kappa_lower_late"])
    & (bayes_nl["kappa_true_late"] <= bayes_nl["kappa_upper_late"])
)
bayes_nl["sigma_coverage"] = (
    (bayes_nl["sigma_true_late"] >= bayes_nl["sigma_lower_late"])
    & (bayes_nl["sigma_true_late"] <= bayes_nl["sigma_upper_late"])
)
bayes_nl["width"] = bayes_nl["kappa_upper_late"] - bayes_nl["kappa_lower_late"]
bayes_nl_calibration = bayes_nl.groupby(["condition", "system"]).agg(
    kappa_coverage=("kappa_coverage", "mean"), sigma_coverage=("sigma_coverage", "mean"),
    width=("width", "median"), n=("repetition", "size"),
).reset_index()
latex_table(
    bayes_nl_calibration,
    "bayes_calibration.tex",
    ["condition", "system", "kappa_coverage", "sigma_coverage", "width", "n"],
    ["Режим", "Система", "Покр. κ", "Покр. σ", "Мед. ширина", "n"],
    {
        "kappa_coverage": lambda x: f"{x:.0%}", "sigma_coverage": lambda x: f"{x:.0%}",
        "width": lambda x: f"{x:.3f}", "n": lambda x: f"{int(x)}",
    },
)
nl_pip = (
    nonlinear.query("scenario == 'stability_loss' and method == 'Bayesian SINDy (spike-slab)'")
    .groupby(["condition", "system"])
    .agg(PIP_x=("pip_x_late", "mean"), PIP_x2=("pip_x2_late", "mean"), PIP_x3=("pip_x3_late", "mean"))
    .reset_index()
)
latex_table(
    nl_pip,
    "bayes_pip.tex",
    ["condition", "system", "PIP_x", "PIP_x2", "PIP_x3"],
    ["Режим", "Система", "PIP(x)", "PIP(x²)", "PIP(x³)"],
    {name: (lambda x: f"{x:.2f}") for name in ["PIP_x", "PIP_x2", "PIP_x3"]},
)

# Main performance figures.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
sns.barplot(ou_summary, x="condition", y="kappa_RMSE", hue="method_short", ax=axes[0])
sns.barplot(ou_summary, x="condition", y="AUC", hue="method_short", ax=axes[1])
axes[0].set_title("OU: ошибка скорости восстановления")
axes[1].set_title("OU: различение механизмов")
for ax in axes:
    ax.set_xlabel("")
    ax.set_xticks(ax.get_xticks(), ["базовый", "стрессовый"])
axes[1].set_ylim(0, 1.03)
axes[0].legend(fontsize=7, title="")
axes[1].get_legend().remove()
plt.tight_layout()
fig.savefig(FIGURES / "ou_benchmark.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, condition in zip(axes, ["baseline", "combined"]):
    matrix = nl_auc.query("condition == @condition").pivot(index="method", columns="system", values="AUC")
    matrix.index = [METHOD_SHORT.get(name, name) for name in matrix.index]
    matrix = matrix.rename(columns=DISPLAY_VALUES["system"])
    sns.heatmap(matrix, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, cbar=ax is axes[-1])
    ax.set_title(f"Нелинейный AUC: {DISPLAY_VALUES['condition'][condition]}")
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.tight_layout()
fig.savefig(FIGURES / "nonlinear_auc.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
plot_calibration = bayes_nl_calibration.copy()
plot_calibration["Режим"] = plot_calibration["condition"].replace(DISPLAY_VALUES["condition"])
plot_calibration["Система"] = plot_calibration["system"].replace(DISPLAY_VALUES["system"])
plot_calibration["label"] = plot_calibration["Режим"] + ": " + plot_calibration["Система"]
sns.barplot(plot_calibration, x="label", y="kappa_coverage", hue="Режим", ax=ax)
ax.axhline(.95, color="black", ls="--", label="номинальные 95%")
ax.set_ylim(0, 1.02)
ax.set_ylabel("Эмпирическое покрытие")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=25)
ax.set_title("Spike-and-slab: калибровка интервалов κ")
plt.tight_layout()
fig.savefig(FIGURES / "bayes_calibration.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Epidemiological summaries from cached predictions.
rolling = pd.read_csv(DATA / "covid_rolling_all_predictions.csv")
rolling_rows = []
for keys, block in rolling.groupby(["architecture", "target", "horizon", "fold"]):
    scale = np.ptp(block["y_true"])
    nrmse = np.sqrt(np.mean((block["prediction"] - block["y_true"]) ** 2)) / scale
    rolling_rows.append((*keys, nrmse))
rolling_metrics = pd.DataFrame(rolling_rows, columns=["architecture", "target", "horizon", "fold", "NRMSE"])
rolling_summary = rolling_metrics.groupby(["architecture", "target", "horizon"]).agg(
    NRMSE_mean=("NRMSE", "mean"), NRMSE_sd=("NRMSE", "std")
).reset_index()
rolling_winners = rolling_summary.loc[
    rolling_summary.groupby(["target", "horizon"])["NRMSE_mean"].idxmin()
]
latex_table(
    rolling_winners.sort_values(["target", "horizon"]),
    "covid_rolling_winners.tex",
    ["target", "horizon", "architecture", "NRMSE_mean", "NRMSE_sd"],
    ["Цель", "Горизонт", "Победитель", "Средний NRMSE", "Ст. откл."],
    {"horizon": lambda x: f"{int(x)} дней", "NRMSE_mean": lambda x: f"{x:.3f}", "NRMSE_sd": lambda x: f"{x:.3f}"},
)

orvi_results = pd.read_csv(DATA / "orvi_classification_results.csv")
latex_table(
    orvi_results.sort_values(["horizon_weeks", "PR-AUC"], ascending=[True, False]),
    "orvi_results.tex",
    ["horizon_weeks", "model", "Precision", "Recall", "F1", "PR-AUC", "n_test"],
    ["h", "Модель", "Точность", "Полнота", "F1", "PR-AUC", "n"],
    {
        "horizon_weeks": lambda x: f"{int(x)} нед.", "Precision": lambda x: f"{x:.3f}",
        "Recall": lambda x: f"{x:.3f}", "F1": lambda x: f"{x:.3f}", "PR-AUC": lambda x: f"{x:.3f}",
        "n_test": lambda x: f"{int(x)}",
    },
)

orvi = pd.read_csv(DATA / "SPb_ORVI_weekly_labelled.csv", parse_dates=["date"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
recent = orvi.query("date >= '2010-01-01'")
axes[0].plot(recent["date"], recent["incidence_per_100k"], color="#315a8a", lw=.8)
axes[0].fill_between(recent["date"], 0, recent["incidence_per_100k"], where=recent["epidemic"].eq(1), color="#c44e52", alpha=.28)
axes[0].set_title("ОРВИ: официальные эпидемические недели")
axes[0].set_ylabel("случаев на 100 тыс.")
rolling_plot = rolling_winners.copy()
rolling_plot["Цель"] = rolling_plot["target"].replace(DISPLAY_VALUES["target"])
sns.barplot(rolling_plot, x="horizon", y="NRMSE_mean", hue="Цель", ax=axes[1])
axes[1].set_title("COVID: лучший NRMSE при скользящем начале")
axes[1].set_xlabel("горизонт, дней")
plt.tight_layout()
fig.savefig(FIGURES / "epidemiology.png", dpi=180, bbox_inches="tight")
plt.close(fig)

best_ou = ou_summary.sort_values(["condition", "AUC"], ascending=[True, False]).groupby("condition").first()
best_nl = nl_average.sort_values(["condition", "AUC"], ascending=[True, False]).groupby("condition").first()
latent_nl = nl_average.query("method == 'Latent Neural SDE (VI)'").set_index("condition")
euler_nl = nl_average.query("method == 'Neural SDE (Euler)'").set_index("condition")
macro_lines = {
    "BestOUBaseline": f"{best_ou.loc['baseline', 'method_short']} ({best_ou.loc['baseline', 'AUC']:.3f})",
    "BestOUCombined": f"{best_ou.loc['combined', 'method_short']} ({best_ou.loc['combined', 'AUC']:.3f})",
    "BestNLBaseline": f"{METHOD_SHORT[best_nl.loc['baseline', 'method']]} ({best_nl.loc['baseline', 'AUC']:.3f})",
    "BestNLCombined": f"{METHOD_SHORT[best_nl.loc['combined', 'method']]} ({best_nl.loc['combined', 'AUC']:.3f})",
    "BayesOUBaseCoverage": f"{100 * bayes_ou_calibration.set_index('condition').loc['baseline', 'kappa_coverage']:.1f}\\%",
    "BayesOUCombinedCoverage": f"{100 * bayes_ou_calibration.set_index('condition').loc['combined', 'kappa_coverage']:.1f}\\%",
    "LatentCombinedAUC": f"{latent_nl.loc['combined', 'AUC']:.3f}",
    "EulerCombinedAUC": f"{euler_nl.loc['combined', 'AUC']:.3f}",
    "LatentCombinedTau": f"{latent_nl.loc['combined', 'tau_RMSE']:.3f}",
}
(REPORT / "generated_metrics.tex").write_text(
    "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macro_lines.items()) + "\n",
    encoding="utf-8",
)
print("Report assets written to", REPORT)
