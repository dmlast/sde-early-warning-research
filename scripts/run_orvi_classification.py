"""Reproduce the labelled ORVI classification table."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sde_early_warning.orvi import (  # noqa: E402
    build_orvi_samples,
    classification_scores,
    fit_with_temporal_validation,
)


def main():
    orvi = pd.read_csv(ROOT / "data/SPb_ORVI_weekly_labelled.csv", parse_dates=["date"])
    ewsnet = pd.read_csv(ROOT / "data/orvi_ewsnet_predictions.csv")
    rows = []
    for horizon in [4, 5]:
        features, _, labels, _ = build_orvi_samples(
            orvi, input_window=30, warning_horizon=horizon
        )
        _, test_result, _, split = fit_with_temporal_validation(features, labels)
        test_result.insert(0, "horizon_weeks", horizon)
        test_result["n_test"] = len(split[2])
        test_result["test_positive"] = int(labels[split[2]].sum())
        rows.append(test_result)

        primary = ewsnet.query(
            "horizon_weeks == @horizon and preprocessing == 'article_features_raw'"
        )
        scores = classification_scores(
            primary["label"].to_numpy(), primary["p_critical_transition"].to_numpy()
        )
        rows.append(
            pd.DataFrame(
                [{
                    "horizon_weeks": horizon,
                    "model": "EWSNet",
                    **scores,
                    "n_test": len(primary),
                    "test_positive": int(primary["label"].sum()),
                }]
            )
        )
    result = pd.concat(rows, ignore_index=True)
    output = ROOT / "data/orvi_classification_results.csv"
    result.to_csv(output, index=False)
    print(result.sort_values(["horizon_weeks", "PR-AUC"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
