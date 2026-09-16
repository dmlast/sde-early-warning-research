"""Feature construction and classifiers for the labelled ORVI time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.ensemble import (
    BalancedBaggingClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier


def trailing_statistic(values, window, statistic):
    """Causal rolling statistic with the article's explicit leading policy."""
    series = pd.Series(values, dtype=float)
    rolling = series.rolling(window=window, min_periods=3)
    result = rolling.var(ddof=1) if statistic == "variance" else rolling.skew()
    return result.bfill().fillna(0.0).to_numpy()


def build_orvi_samples(frame, input_window=30, warning_horizon=4):
    """Implement the article's equations (1)-(5) without epidemic leakage."""
    values = frame["incidence_per_100k"].to_numpy(float)
    epidemic = frame["epidemic"].to_numpy(int)
    dates = pd.to_datetime(frame["date"]).to_numpy()
    records = []
    segment_start = 0
    for index in range(1, len(frame) + 1):
        segment_finished = index == len(frame) or epidemic[index] != epidemic[index - 1]
        if not segment_finished:
            continue
        if epidemic[index - 1] == 0:
            for end in range(segment_start + input_window - 1, index):
                start = end - input_window + 1
                raw_window = values[start : end + 1]
                future_epidemic = np.flatnonzero(epidemic[end + 1 :] == 1)
                weeks_to_onset = (
                    int(future_epidemic[0] + 1) if len(future_epidemic) else np.inf
                )
                channels = [
                    trailing_statistic(raw_window, 20, "skew"),
                    trailing_statistic(raw_window, 15, "variance"),
                    trailing_statistic(raw_window, 20, "variance"),
                    trailing_statistic(raw_window, 30, "variance"),
                ]
                records.append(
                    {
                        "end_date": pd.Timestamp(dates[end]),
                        "weeks_to_onset": weeks_to_onset,
                        "label": int(weeks_to_onset <= warning_horizon),
                        "features": np.concatenate(channels),
                        "raw_window": raw_window.copy(),
                    }
                )
        segment_start = index
    features = np.stack([record["features"] for record in records])
    raw_windows = np.stack([record["raw_window"] for record in records])
    labels = np.array([record["label"] for record in records], dtype=int)
    metadata = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key not in {"features", "raw_window"}
            }
            for record in records
        ]
    )
    return features, raw_windows, labels, metadata


def chronological_split(n_samples):
    train_end = int(np.ceil(0.70 * n_samples))
    validation_end = train_end + int(np.ceil(0.10 * n_samples))
    return (
        np.arange(0, train_end),
        np.arange(train_end, validation_end),
        np.arange(validation_end, n_samples),
    )


def candidate_classifiers(seed=20260911):
    """Finite validation grid; test observations never select hyperparameters."""
    return {
        "Decision Tree": [
            DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight="balanced",
                random_state=seed,
            )
            for depth in [3, 5, None]
            for leaf in [2, 5, 10]
        ],
        "Random Forest": [
            RandomForestClassifier(
                n_estimators=300,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=seed,
            )
            for depth in [5, None]
            for leaf in [2, 5]
        ],
        "Easy Ensemble": [
            EasyEnsembleClassifier(n_estimators=n, random_state=seed, n_jobs=1)
            for n in [10, 25]
        ],
        "RUSBoost": [
            RUSBoostClassifier(
                n_estimators=n,
                learning_rate=rate,
                random_state=seed,
            )
            for n in [50, 150]
            for rate in [0.05, 0.2, 1.0]
        ],
        "Balanced Bagging": [
            BalancedBaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=depth),
                n_estimators=100,
                random_state=seed,
                n_jobs=1,
            )
            for depth in [3, 5, None]
        ],
    }


def positive_probability(model, features):
    probabilities = model.predict_proba(features)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    return probabilities[:, positive_column]


def classification_scores(y_true, probability, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "PR-AUC": average_precision_score(y_true, probability),
    }


def fit_with_temporal_validation(features, labels, seed=20260911):
    """Select on validation, refit on train+validation and score test once."""
    train_idx, validation_idx, test_idx = chronological_split(len(labels))
    fitted = {}
    validation_rows = []
    test_rows = []
    for name, candidates in candidate_classifiers(seed).items():
        candidate_results = []
        for candidate in candidates:
            candidate.fit(features[train_idx], labels[train_idx])
            probability = positive_probability(candidate, features[validation_idx])
            score = classification_scores(labels[validation_idx], probability)
            candidate_results.append((score["PR-AUC"], score["F1"], candidate))
        _, _, best = max(candidate_results, key=lambda item: (item[0], item[1]))
        final_model = clone(best).fit(
            features[np.r_[train_idx, validation_idx]],
            labels[np.r_[train_idx, validation_idx]],
        )
        validation_probability = positive_probability(best, features[validation_idx])
        test_probability = positive_probability(final_model, features[test_idx])
        validation_rows.append(
            {
                "model": name,
                **classification_scores(labels[validation_idx], validation_probability),
            }
        )
        test_rows.append(
            {"model": name, **classification_scores(labels[test_idx], test_probability)}
        )
        fitted[name] = final_model
    return (
        pd.DataFrame(validation_rows),
        pd.DataFrame(test_rows),
        fitted,
        (train_idx, validation_idx, test_idx),
    )
