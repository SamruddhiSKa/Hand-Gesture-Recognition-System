"""Controlled in-memory classifier comparison for normalized hand landmarks."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "landmark_dataset.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "classifier_comparison.json"
EXPECTED_DATASET_SHA256 = "29c682b9bd6d2fa63417edcb745ad5df6a188fc3e84e9a954a1eb43900ee1ca9"
EXPECTED_ROWS = 3263
EXPECTED_FEATURES = 42
EXPECTED_CLASSES = 36
EXPECTED_TRAIN_ROWS = 2610
EXPECTED_TEST_ROWS = 653
TEST_SIZE = 0.2
RANDOM_STATE = 42
TIMING_REPEATS = 5


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_indices(indices: np.ndarray) -> str:
    return sha256_bytes(np.asarray(indices, dtype="<i8").tobytes())


def build_estimators() -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "rbf_svm": Pipeline(
            [
                ("standard_scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight=None,
                        probability=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            [
                ("standard_scaler", StandardScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="uniform",
                        metric="minkowski",
                        p=2,
                    ),
                ),
            ]
        ),
    }


def describe_settings(name: str, estimator: Any) -> dict[str, Any]:
    if name == "random_forest":
        return {
            "estimator": "RandomForestClassifier",
            "n_estimators": estimator.n_estimators,
            "random_state": estimator.random_state,
        }

    scaler = estimator.named_steps["standard_scaler"]
    if name == "rbf_svm":
        svc = estimator.named_steps["svc"]
        return {
            "pipeline": ["StandardScaler", "SVC"],
            "standard_scaler": scaler.get_params(),
            "svc": {
                "kernel": svc.kernel,
                "C": svc.C,
                "gamma": svc.gamma,
                "class_weight": svc.class_weight,
                "probability": svc.probability,
                "random_state": svc.random_state,
            },
        }

    knn = estimator.named_steps["knn"]
    return {
        "pipeline": ["StandardScaler", "KNeighborsClassifier"],
        "standard_scaler": scaler.get_params(),
        "knn": {
            "n_neighbors": knn.n_neighbors,
            "weights": knn.weights,
            "metric": knn.metric,
            "p": knn.p,
        },
    }


def measure_classifier_latency(
    estimator: Any, features: pd.DataFrame
) -> tuple[np.ndarray, list[float]]:
    """Measure warm one-row predict calls; leave row slicing outside the timer."""
    estimator.predict(features.iloc[:1])
    predictions: np.ndarray | None = None
    timings_ms: list[float] = []

    for repeat in range(TIMING_REPEATS):
        for row_index in range(len(features)):
            row = features.iloc[[row_index]]
            started = time.perf_counter_ns()
            prediction = estimator.predict(row)[0]
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            timings_ms.append(elapsed_ms)
            if repeat == 0:
                if predictions is None:
                    predictions = np.empty(len(features), dtype=object)
                predictions[row_index] = prediction

    if predictions is None:
        raise RuntimeError("No test predictions were produced")
    return predictions, timings_ms


def run() -> dict[str, Any]:
    if RESULTS_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {RESULTS_PATH}")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Expected dataset not found: {DATASET_PATH}")

    dataset_bytes = DATASET_PATH.read_bytes()
    dataset_sha256 = sha256_bytes(dataset_bytes)
    if dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "Dataset fingerprint differs from the approved preflight; "
            "no classifier has been fitted."
        )

    data = pd.read_csv(DATASET_PATH)
    if "label" not in data.columns:
        raise ValueError("Dataset is missing the required 'label' column")
    features = data.drop(columns=["label"])
    labels = data["label"]
    if (
        len(data) != EXPECTED_ROWS
        or features.shape[1] != EXPECTED_FEATURES
        or labels.nunique() != EXPECTED_CLASSES
        or not all(pd.api.types.is_numeric_dtype(features[column]) for column in features)
        or not np.isfinite(features.to_numpy()).all()
    ):
        raise ValueError(
            "Dataset dimensions, classes, numeric feature types, or finite-value "
            "checks differ from the approved preflight; no classifier has been fitted."
        )

    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    if (len(train_indices), len(test_indices)) != (
        EXPECTED_TRAIN_ROWS,
        EXPECTED_TEST_ROWS,
    ):
        raise ValueError(
            "Split sizes differ from the approved 2610/653 plan; "
            "no classifier has been fitted."
        )

    X_train = features.iloc[train_indices]
    X_test = features.iloc[test_indices]
    y_train = labels.iloc[train_indices]
    y_test = labels.iloc[test_indices]
    class_labels = sorted(labels.unique().tolist())

    estimators = build_estimators()
    outcomes: dict[str, Any] = {}
    for name, estimator in estimators.items():
        outcome: dict[str, Any] = {
            "status": "running",
            "settings": describe_settings(name, estimator),
            "warnings": [],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                estimator.fit(X_train, y_train)
                predictions, timings_ms = measure_classifier_latency(
                    estimator, X_test
                )
                report = classification_report(
                    y_test,
                    predictions,
                    labels=class_labels,
                    output_dict=True,
                    zero_division=0,
                )
                outcome.update(
                    {
                        "status": "success",
                        "metrics": {
                            "accuracy": float(accuracy_score(y_test, predictions)),
                            "macro_f1": float(report["macro avg"]["f1-score"]),
                            "weighted_f1": float(report["weighted avg"]["f1-score"]),
                            "per_class": {
                                str(label): {
                                    "precision": float(report[str(label)]["precision"]),
                                    "recall": float(report[str(label)]["recall"]),
                                    "f1_score": float(report[str(label)]["f1-score"]),
                                    "support": int(report[str(label)]["support"]),
                                }
                                for label in class_labels
                            },
                            "confusion_matrix_labels": [
                                str(label) for label in class_labels
                            ],
                            "confusion_matrix": confusion_matrix(
                                y_test, predictions, labels=class_labels
                            ).tolist(),
                        },
                        "classifier_only_latency": {
                            "measurement": (
                                "Warm estimator.predict() on one held-out row per "
                                "call; row selection excluded; pipeline transforms "
                                "included where applicable."
                            ),
                            "warmup_calls": 1,
                            "repeats": TIMING_REPEATS,
                            "timing_samples": len(timings_ms),
                            "mean_ms": float(np.mean(timings_ms)),
                            "median_ms": float(np.percentile(timings_ms, 50)),
                            "p95_ms": float(np.percentile(timings_ms, 95)),
                            "min_ms": float(np.min(timings_ms)),
                            "max_ms": float(np.max(timings_ms)),
                        },
                    }
                )
            except Exception as exc:
                outcome.update(
                    {
                        "status": "failed",
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    }
                )
            finally:
                outcome["warnings"] = [
                    {
                        "category": warning.category.__name__,
                        "message": str(warning.message),
                    }
                    for warning in caught
                ]
        outcomes[name] = outcome

    result = {
        "experiment": "Controlled classifier comparison on existing normalized landmarks",
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": dataset_sha256,
            "rows": int(len(data)),
            "feature_count": int(features.shape[1]),
            "class_count": int(labels.nunique()),
            "class_counts": {
                str(label): int(count)
                for label, count in labels.value_counts().sort_index().items()
            },
            "features_are_existing_normalized_values": True,
        },
        "split": {
            "method": "train_test_split",
            "test_size": TEST_SIZE,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_indices_sha256": sha256_indices(train_indices),
            "test_indices_sha256": sha256_indices(test_indices),
            "same_train_and_test_indices_for_all_classifiers": True,
            "test_set_used_for_tuning": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "models": outcomes,
        "restrictions_observed": {
            "existing_model_loaded": False,
            "existing_model_modified_or_overwritten": False,
            "images_or_videos_processed": False,
            "hyperparameter_tuning_performed": False,
        },
    }

    with RESULTS_PATH.open("x", encoding="utf-8", newline="\n") as result_file:
        json.dump(result, result_file, indent=2, allow_nan=False)
        result_file.write("\n")
    return result


if __name__ == "__main__":
    completed = run()
    print(f"Results saved: {RESULTS_PATH}")
    print(
        json.dumps(
            {
                name: {
                    "status": outcome["status"],
                    **(
                        outcome.get("metrics", {})
                        if outcome["status"] == "success"
                        else {"error": outcome.get("error")}
                    ),
                }
                for name, outcome in completed["models"].items()
            },
            indent=2,
        )
    )
