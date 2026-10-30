"""Measure baseline classifier-only latency with the specified warm-up count."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "landmark_dataset.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXPECTED_SHA256 = "29c682b9bd6d2fa63417edcb745ad5df6a188fc3e84e9a954a1eb43900ee1ca9"
WARMUP_CALLS = 50
TIMED_REPEATS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42


def hardware_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "cpu_model": None,
        "installed_ram_gib": None,
        "query_error": None,
    }
    command = (
        "$cpu=(Get-CimInstance Win32_Processor | Select-Object -First 1 "
        "-ExpandProperty Name); "
        "$ram=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
        "[pscustomobject]@{cpu_model=$cpu; installed_ram_bytes=$ram} | "
        "ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        details["query_error"] = f"{type(exc).__name__}: {exc}"
        return details
    queried = json.loads(completed.stdout)
    details["cpu_model"] = queried["cpu_model"]
    details["installed_ram_gib"] = int(queried["installed_ram_bytes"]) / (1024**3)
    return details


def main() -> None:
    if RESULTS_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {RESULTS_DIR}")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Expected existing dataset is unavailable: {DATASET_PATH}")

    dataset_bytes = DATASET_PATH.read_bytes()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    if dataset_sha256 != EXPECTED_SHA256:
        raise ValueError("Dataset fingerprint differs from the audited dataset.")

    data = pd.read_csv(DATASET_PATH)
    if "label" not in data.columns:
        raise ValueError("Dataset is missing its expected label column.")
    X = data.drop(columns=["label"])
    y = data["label"]
    if len(data) != 3263 or X.shape[1] != 42 or y.nunique() != 36:
        raise ValueError("Dataset dimensions or class count differ from the audit.")

    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    if (len(train_indices), len(test_indices)) != (2610, 653):
        raise ValueError("Expected 2,610 training and 653 test rows.")
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train = y.iloc[train_indices]

    estimators = {
        "random_forest": Pipeline(
            [("classifier", RandomForestClassifier(n_estimators=100, random_state=42))]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=5, weights="uniform", metric="minkowski", p=2
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=False,
                        class_weight=None,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    rows: list[dict[str, Any]] = []
    captured_warnings: list[dict[str, str]] = []
    for name, estimator in estimators.items():
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            estimator.fit(X_train, y_train)
            for warm_index in range(WARMUP_CALLS):
                estimator.predict(X_test.iloc[[warm_index % len(X_test)]])

            elapsed_ms: list[float] = []
            for _ in range(TIMED_REPEATS):
                for row_index in range(len(X_test)):
                    row = X_test.iloc[[row_index]]
                    start = time.perf_counter()
                    estimator.predict(row)
                    elapsed_ms.append((time.perf_counter() - start) * 1000)

        captured_warnings.extend(
            {
                "model": name,
                "category": item.category.__name__,
                "message": str(item.message),
            }
            for item in captured
        )
        rows.append(
            {
                "model": name,
                "median_ms": float(np.percentile(elapsed_ms, 50)),
                "p95_ms": float(np.percentile(elapsed_ms, 95)),
                "mean_ms": float(np.mean(elapsed_ms)),
                "warmup_calls": WARMUP_CALLS,
                "timed_calls": len(elapsed_ms),
                "repeats": TIMED_REPEATS,
                "timed_rows_per_repeat": len(X_test),
                "scaler_included": name in {"knn", "rbf_svm"},
                "timed_operation": "single-row pipeline.predict",
            }
        )

    RESULTS_DIR.mkdir()
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "latency.csv", index=False)
    result = {
        "experiment": "Corrected baseline classifier-only latency measurement",
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": dataset_sha256,
            "rows": len(data),
            "features": X.shape[1],
            "classes": int(y.nunique()),
        },
        "split": {
            "method": "stratified train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "train_indices_sha256": hashlib.sha256(
                np.asarray(train_indices, dtype="<i8").tobytes()
            ).hexdigest(),
            "test_indices_sha256": hashlib.sha256(
                np.asarray(test_indices, dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "method": (
            "Each fixed baseline is fitted on the training partition; 50 warm-up "
            "single-row predictions; then five passes over all 653 test rows "
            "timed individually with time.perf_counter. Row slicing excluded. "
            "KNN and SVM scaling included in Pipeline.predict. Camera, MediaPipe, "
            "feature extraction, rendering, and UI excluded."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
            "hardware": hardware_details(),
        },
        "settings": {
            "random_forest": {"n_estimators": 100, "random_state": 42},
            "knn": {
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "euclidean",
                "scaler": "StandardScaler fitted on training rows",
            },
            "rbf_svm": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "scaler": "StandardScaler fitted on training rows",
            },
        },
        "latency": {"rows": rows, "warnings": captured_warnings},
        "errors": [],
    }
    (RESULTS_DIR / "latency.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"{row['model']}: median={row['median_ms']:.4f} ms; "
            f"P95={row['p95_ms']:.4f} ms; calls={row['timed_calls']}; "
            f"warm-ups={row['warmup_calls']}; scaler={row['scaler_included']}"
        )
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
