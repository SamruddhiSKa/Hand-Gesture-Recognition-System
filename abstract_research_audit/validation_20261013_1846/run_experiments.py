"""Run controlled classifier validation using the existing normalized CSV."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "landmark_dataset.csv"
RESULTS_DIR = RUN_ROOT / "results"
EXPECTED_SHA256 = "29c682b9bd6d2fa63417edcb745ad5df6a188fc3e84e9a954a1eb43900ee1ca9"
EXPECTED_ROWS = 3263
EXPECTED_FEATURES = 42
EXPECTED_CLASSES = 36
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
CV_REPEATS = 10
SEARCH_FOLDS = 5
LATENCY_WARMUPS = 50
LATENCY_REPEATS = 5


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def index_hash(indices: np.ndarray) -> str:
    return sha256(np.asarray(indices, dtype="<i8").tobytes())


def pipeline_estimators() -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="uniform",
                        metric="minkowski",
                        p=2,
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
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def parameter_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 20, 40],
        },
        "knn": {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
        },
        "rbf_svm": {
            "classifier__C": [1, 10, 100],
            "classifier__gamma": ["scale", 0.01, 0.1],
        },
    }


def hardware_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor_platform_string": platform.processor(),
        "cpu_model": None,
        "installed_ram_gib": None,
        "query_error": None,
    }
    if platform.system() != "Windows":
        details["query_error"] = (
            "CPU model and installed RAM query is implemented only for Windows."
        )
        return details

    command = (
        "$cpu=(Get-CimInstance Win32_Processor | Select-Object -First 1 "
        "-ExpandProperty Name); "
        "$ram=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
        "[pscustomobject]@{cpu_model=$cpu; installed_ram_bytes=$ram} | "
        "ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        queried = json.loads(completed.stdout)
        details["cpu_model"] = queried["cpu_model"]
        details["installed_ram_bytes"] = int(queried["installed_ram_bytes"])
        details["installed_ram_gib"] = (
            details["installed_ram_bytes"] / (1024**3)
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        details["query_error"] = f"{type(exc).__name__}: {exc}"
    return details


def classify_feature_and_sequence_gaps() -> dict[str, str]:
    return {
        "normalization_ablation": (
            "Not run: the available CSV stores only wrist-relative, "
            "max-absolute-scale-normalized features. Raw coordinates and the "
            "pre-scale wrist-relative vectors cannot be recovered by inversion. "
            "The extraction code accepts image dimensions for aspect correction, "
            "but those original per-sample dimensions are not in the CSV. "
            "No source images were processed."
        ),
        "temporal_smoothing": (
            "Not run: no labeled video/ordered sequence artifact was found in "
            "the pre-execution read-only audit. The available landmark table "
            "contains feature columns and label only, with no sequence/frame "
            "order or timestamps suitable for sequence-level evaluation."
        ),
        "signer_session_generalization": (
            "Not run: the available landmark table contains no signer or session "
            "identifier, and no valid group-held-out split can be formed."
        ),
    }


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(RESULTS_DIR / filename, index=False, encoding="utf-8")


def write_summary(result: dict[str, Any]) -> None:
    lines = [
        "# Research validation run",
        "",
        "## Design and integrity checks",
        "",
        f"- Dataset SHA-256: `{result['dataset']['sha256']}`",
        f"- Rows/features/classes: {result['dataset']['rows']} / "
        f"{result['dataset']['feature_count']} / {result['dataset']['class_count']}",
        f"- Holdout split: stratified 80:20, seed {RANDOM_STATE}; "
        f"{result['holdout']['train_rows']} train / "
        f"{result['holdout']['test_rows']} test.",
        "- Repeated CV runs only on the 80% training partition so the fixed "
        "holdout remains outside CV and all grid-search model selection.",
        "- No estimator or scaler is fitted on the test partition.",
        "- No existing serialized model is loaded or saved.",
        "",
        "## Repeated stratified CV on training partition",
        "",
        "| Estimator | Accuracy (mean ± SD) | Macro F1 (mean ± SD) | "
        "Weighted F1 (mean ± SD) | Fold wins by macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    cv_summary = result.get("repeated_cv", {}).get("summary", [])
    for row in cv_summary:
        lines.append(
            f"| {row['model']} | {row['accuracy_mean']:.4f} ± "
            f"{row['accuracy_std']:.4f} | {row['macro_f1_mean']:.4f} ± "
            f"{row['macro_f1_std']:.4f} | {row['weighted_f1_mean']:.4f} ± "
            f"{row['weighted_f1_std']:.4f} | {row['macro_f1_fold_wins']} |"
        )
    lines.extend(
        [
            "",
            "Fold-wise wins compare scores only within the same held-out fold. "
            "They do not establish signer/session-independent generalization.",
            "",
            "## Grid search and one-time holdout evaluation",
            "",
            "| Estimator | Best training-CV macro F1 | Best parameters | "
            "Holdout accuracy | Holdout macro F1 | Holdout weighted F1 |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in result.get("tuned_holdout_evaluation", {}).get(
        "per_model_summary", []
    ):
        params = json.dumps(row["best_params"], sort_keys=True)
        lines.append(
            f"| {row['model']} | {row['best_cv_macro_f1']:.4f} | "
            f"`{params}` | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['weighted_f1']:.4f} |"
        )
    best_name = result.get("tuned_holdout_evaluation", {}).get(
        "best_model_by_training_cv_macro_f1"
    )
    if best_name:
        lines.extend(
            [
                "",
                f"Best configured estimator selected by training-only grid-search "
                f"CV macro F1: **{best_name}**. The fixed test set did not select "
                "the model.",
            ]
        )
    lines.extend(
        [
            "",
            "## Baseline estimator latency",
            "",
            "| Estimator | Median ms | P95 ms | Timed calls | Scaling included? |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in result.get("latency", {}).get("rows", []):
        lines.append(
            f"| {row['model']} | {row['median_ms']:.4f} | {row['p95_ms']:.4f} | "
            f"{row['timed_calls']} | "
            f"{'yes, in pipeline' if row['scaler_included'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Timing is warm single-row `predict()` latency: row slicing excluded; "
            "scaler transforms included for KNN/SVM; camera, MediaPipe, rendering, "
            "and UI latency excluded.",
            "",
            "## Unrun questions / limitations",
            "",
        ]
    )
    for key, text in result["unavailable_experiments"].items():
        lines.append(f"- **{key}:** {text}")
    lines.extend(
        [
            "",
            "The training/live aspect-correction discrepancy remains separate and "
            "unmodified: extraction operates on square still crops and passes "
            "`image.shape`, while live inference passes full-frame dimensions to "
            "the aspect-correction function. This run does not test or correct it.",
            "",
            "## Warnings and failures",
            "",
            f"- CV: "
            f"{len(result.get('repeated_cv', {}).get('warnings', []))} warning(s).",
            f"- Grid search/model evaluation: "
            f"{len(result.get('grid_search', {}).get('warnings', []))} warning(s).",
            f"- Latency fits: "
            f"{len(result.get('latency', {}).get('warnings', []))} warning(s).",
            "- Per-model errors and warning details are recorded in "
            "`experiment_results.json`.",
            "",
        ]
    )
    summary_path = RESULTS_DIR / "summary.md"
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def run_experiments() -> dict[str, Any]:
    if RESULTS_DIR.exists():
        raise FileExistsError(
            f"Refusing to use existing results directory: {RESULTS_DIR}"
        )
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset unavailable: {DATASET_PATH}")

    data_bytes = DATASET_PATH.read_bytes()
    dataset_digest = sha256(data_bytes)
    if dataset_digest != EXPECTED_SHA256:
        raise ValueError(
            "Dataset fingerprint differs from the audited file; stopping "
            "before any model fitting."
        )

    data = pd.read_csv(DATASET_PATH)
    if "label" not in data.columns:
        raise ValueError("Dataset lacks the expected label column")
    X = data.drop(columns=["label"])
    y = data["label"]
    if (
        len(data) != EXPECTED_ROWS
        or X.shape[1] != EXPECTED_FEATURES
        or y.nunique() != EXPECTED_CLASSES
        or not all(pd.api.types.is_numeric_dtype(X[column]) for column in X)
        or not np.isfinite(X.to_numpy()).all()
    ):
        raise ValueError(
            "Dataset dimensions, class count, feature types, or finite-value "
            "checks differ; stopping before model fitting."
        )

    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    if (len(train_indices), len(test_indices)) != (2610, 653):
        raise ValueError(
            "Expected 2610/653 split sizes; stopping before model fitting."
        )
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    labels = sorted(y.unique().tolist())

    estimators = pipeline_estimators()
    cv_splitter = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    cv_splits = list(cv_splitter.split(X_train, y_train))
    if len(cv_splits) != CV_FOLDS * CV_REPEATS:
        raise RuntimeError("Repeated CV did not produce the expected 50 folds")

    cv_fold_rows: list[dict[str, Any]] = []
    cv_warnings: list[dict[str, str]] = []
    cv_errors: dict[str, dict[str, str]] = {}
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }
    for name, estimator in estimators.items():
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                scores = cross_validate(
                    estimator,
                    X_train,
                    y_train,
                    cv=cv_splits,
                    scoring=scoring,
                    n_jobs=1,
                    error_score="raise",
                    return_train_score=False,
                )
            cv_warnings.extend(
                {
                    "model": name,
                    "category": item.category.__name__,
                    "message": str(item.message),
                }
                for item in captured
            )
            for fold_index in range(len(cv_splits)):
                repeat_index, within_repeat_fold = divmod(fold_index, CV_FOLDS)
                fold_train, fold_valid = cv_splits[fold_index]
                cv_fold_rows.append(
                    {
                        "model": name,
                        "repeat": repeat_index + 1,
                        "fold": within_repeat_fold + 1,
                        "train_rows": len(fold_train),
                        "validation_rows": len(fold_valid),
                        "train_indices_sha256": index_hash(fold_train),
                        "validation_indices_sha256": index_hash(fold_valid),
                        "accuracy": float(scores["test_accuracy"][fold_index]),
                        "macro_f1": float(scores["test_macro_f1"][fold_index]),
                        "weighted_f1": float(scores["test_weighted_f1"][fold_index]),
                    }
                )
        except Exception as exc:
            cv_errors[name] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

    cv_summary: list[dict[str, Any]] = []
    if cv_fold_rows:
        folds_frame = pd.DataFrame(cv_fold_rows)
        for name in estimators:
            model_rows = folds_frame[folds_frame["model"] == name]
            if model_rows.empty:
                continue
            cv_summary.append(
                {
                    "model": name,
                    "successful_folds": int(len(model_rows)),
                    "accuracy_mean": float(model_rows["accuracy"].mean()),
                    "accuracy_std": float(model_rows["accuracy"].std(ddof=1)),
                    "macro_f1_mean": float(model_rows["macro_f1"].mean()),
                    "macro_f1_std": float(model_rows["macro_f1"].std(ddof=1)),
                    "weighted_f1_mean": float(model_rows["weighted_f1"].mean()),
                    "weighted_f1_std": float(model_rows["weighted_f1"].std(ddof=1)),
                    "macro_f1_fold_wins": 0,
                    "accuracy_fold_wins": 0,
                }
            )
        if len(cv_summary) > 1:
            cv_pivot = folds_frame.pivot(
                index=["repeat", "fold"], columns="model", values="macro_f1"
            )
            acc_pivot = folds_frame.pivot(
                index=["repeat", "fold"], columns="model", values="accuracy"
            )
            macro_winners = cv_pivot.idxmax(axis=1)
            accuracy_winners = acc_pivot.idxmax(axis=1)
            for row in cv_summary:
                row["macro_f1_fold_wins"] = int(
                    (macro_winners == row["model"]).sum()
                )
                row["accuracy_fold_wins"] = int(
                    (accuracy_winners == row["model"]).sum()
                )

    search_cv = StratifiedKFold(
        n_splits=SEARCH_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    tuned_models: dict[str, Any] = {}
    tuning_rows: list[dict[str, Any]] = []
    tuned_test_rows: list[dict[str, Any]] = []
    tuned_per_class_rows: list[dict[str, Any]] = []
    tuned_confusion_rows: list[dict[str, Any]] = []
    search_warnings: list[dict[str, str]] = []
    search_errors: dict[str, dict[str, str]] = {}
    for name, estimator in estimators.items():
        try:
            search = GridSearchCV(
                estimator,
                param_grid=parameter_grids()[name],
                scoring="f1_macro",
                cv=search_cv,
                refit=True,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                search.fit(X_train, y_train)
            search_warnings.extend(
                {
                    "model": name,
                    "category": item.category.__name__,
                    "message": str(item.message),
                }
                for item in captured
            )
            tuned_models[name] = search.best_estimator_
            for index, params in enumerate(search.cv_results_["params"]):
                tuning_rows.append(
                    {
                        "model": name,
                        "params": json.dumps(params, sort_keys=True),
                        "mean_cv_macro_f1": float(
                            search.cv_results_["mean_test_score"][index]
                        ),
                        "std_cv_macro_f1": float(
                            search.cv_results_["std_test_score"][index]
                        ),
                        "rank": int(search.cv_results_["rank_test_score"][index]),
                    }
                )

            predictions = search.best_estimator_.predict(X_test)
            report = classification_report(
                y_test,
                predictions,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
            tuned_test_rows.append(
                {
                    "model": name,
                    "best_params": search.best_params_,
                    "best_cv_macro_f1": float(search.best_score_),
                    "accuracy": float(accuracy_score(y_test, predictions)),
                    "macro_f1": float(report["macro avg"]["f1-score"]),
                    "weighted_f1": float(report["weighted avg"]["f1-score"]),
                    "test_rows": int(len(y_test)),
                }
            )
            for label in labels:
                label_report = report[str(label)]
                tuned_per_class_rows.append(
                    {
                        "model": name,
                        "class": label,
                        "precision": float(label_report["precision"]),
                        "recall": float(label_report["recall"]),
                        "f1_score": float(label_report["f1-score"]),
                        "support": int(label_report["support"]),
                    }
                )
            matrix = confusion_matrix(y_test, predictions, labels=labels)
            for actual_index, actual in enumerate(labels):
                for predicted_index, predicted in enumerate(labels):
                    tuned_confusion_rows.append(
                        {
                            "model": name,
                            "actual": actual,
                            "predicted": predicted,
                            "count": int(matrix[actual_index, predicted_index]),
                        }
                    )
        except Exception as exc:
            search_errors[name] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

    best_tuned_name: str | None = None
    best_tuned_weakest: list[dict[str, Any]] = []
    best_tuned_confusions: list[dict[str, Any]] = []
    if tuned_test_rows:
        best_tuned_name = max(
            tuned_test_rows, key=lambda row: row["best_cv_macro_f1"]
        )["model"]
        best_classes = [
            row
            for row in tuned_per_class_rows
            if row["model"] == best_tuned_name
        ]
        best_tuned_weakest = sorted(
            best_classes, key=lambda row: (row["f1_score"], row["class"])
        )[:3]
        best_tuned_confusions = sorted(
            (
                row
                for row in tuned_confusion_rows
                if row["model"] == best_tuned_name
                and row["actual"] != row["predicted"]
                and row["count"] > 0
            ),
            key=lambda row: (-row["count"], row["actual"], row["predicted"]),
        )[:5]

    latency_rows: list[dict[str, Any]] = []
    latency_warnings: list[dict[str, str]] = []
    latency_errors: dict[str, dict[str, str]] = {}
    for name, estimator in estimators.items():
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                estimator.fit(X_train, y_train)
                for warm_index in range(LATENCY_WARMUPS):
                    estimator.predict(X_test.iloc[[warm_index % len(X_test)]])
                elapsed_ms: list[float] = []
                for _ in range(LATENCY_REPEATS):
                    for row_index in range(len(X_test)):
                        row = X_test.iloc[[row_index]]
                        started = time.perf_counter()
                        estimator.predict(row)
                        elapsed_ms.append((time.perf_counter() - started) * 1000)
            latency_warnings.extend(
                {
                    "model": name,
                    "category": item.category.__name__,
                    "message": str(item.message),
                }
                for item in captured
            )
            latency_rows.append(
                {
                    "model": name,
                    "median_ms": float(np.percentile(elapsed_ms, 50)),
                    "p95_ms": float(np.percentile(elapsed_ms, 95)),
                    "mean_ms": float(np.mean(elapsed_ms)),
                    "warmup_calls": LATENCY_WARMUPS,
                    "timed_calls": len(elapsed_ms),
                    "repeats": LATENCY_REPEATS,
                    "timed_rows_per_repeat": int(len(X_test)),
                    "scaler_included": name in {"knn", "rbf_svm"},
                    "timed_operation": "single-row pipeline.predict",
                }
            )
        except Exception as exc:
            latency_errors[name] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

    result = {
        "experiment": "Repeated CV, training-only tuning, and latency verification",
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": dataset_digest,
            "rows": int(len(data)),
            "feature_count": int(X.shape[1]),
            "class_count": int(y.nunique()),
            "class_counts": {
                str(label): int(count)
                for label, count in y.value_counts().sort_index().items()
            },
            "features_are_existing_normalized_values": True,
        },
        "holdout": {
            "procedure": "Stratified 80:20 split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_indices_sha256": index_hash(train_indices),
            "test_indices_sha256": index_hash(test_indices),
            "test_used_for_tuning": False,
            "note": (
                "The held-out partition was reserved for one evaluation of each "
                "tuned best estimator. Repeated CV and grid search used only the "
                "training partition."
            ),
        },
        "repeated_cv": {
            "scope": "80% training partition only",
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": CV_FOLDS,
            "n_repeats": CV_REPEATS,
            "random_state": RANDOM_STATE,
            "scoring": ["accuracy", "f1_macro", "f1_weighted"],
            "shared_fold_indices": True,
            "std_ddof": 1,
            "fold_rows": cv_fold_rows,
            "summary": cv_summary,
            "errors": cv_errors,
            "warnings": cv_warnings,
        },
        "grid_search": {
            "scope": "80% training partition only",
            "splitter": "StratifiedKFold",
            "n_splits": SEARCH_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "scoring": "f1_macro",
            "parameter_grids": parameter_grids(),
            "candidate_scores": tuning_rows,
            "best_models_selected_by_training_cv_macro_f1": [
                {
                    "model": row["model"],
                    "best_params": row["best_params"],
                    "best_cv_macro_f1": row["best_cv_macro_f1"],
                }
                for row in tuned_test_rows
            ],
            "errors": search_errors,
            "warnings": search_warnings,
        },
        "tuned_holdout_evaluation": {
            "per_model_summary": tuned_test_rows,
            "per_class_metrics": tuned_per_class_rows,
            "confusion_matrix_long": tuned_confusion_rows,
            "best_model_by_training_cv_macro_f1": best_tuned_name,
            "best_model_weakest_three_classes": best_tuned_weakest,
            "best_model_top_five_confusions": best_tuned_confusions,
            "errors": search_errors,
        },
        "latency": {
            "method": (
                "Fit each fixed baseline on the training partition; 50 warm-up "
                "single-row predict calls; then five passes over the 653 test "
                "rows timed individually with time.perf_counter. Row slicing "
                "excluded. KNN/SVM pipeline scaler included."
            ),
            "camera_mediapipe_display_included": False,
            "rows": latency_rows,
            "errors": latency_errors,
            "warnings": latency_warnings,
        },
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
        "unavailable_experiments": classify_feature_and_sequence_gaps(),
        "preprocessing_mismatch": (
            "Not corrected or evaluated: training extraction operates on square "
            "still-image crops and passes image.shape; live inference passes full "
            "frame dimensions to x aspect correction. A non-square negotiated "
            "frame may differ from training feature scaling."
        ),
        "safeguards": {
            "existing_model_loaded": False,
            "existing_model_saved_or_overwritten": False,
            "existing_dataset_modified": False,
            "images_videos_or_webcam_accessed": False,
            "test_used_for_hyperparameter_selection": False,
        },
    }

    RESULTS_DIR.mkdir(parents=False, exist_ok=False)
    save_csv(pd.DataFrame(cv_fold_rows), "repeated_cv_fold_scores.csv")
    save_csv(pd.DataFrame(cv_summary), "repeated_cv_summary.csv")
    save_csv(pd.DataFrame(tuning_rows), "grid_search_candidate_scores.csv")
    save_csv(pd.DataFrame(tuned_test_rows), "tuned_holdout_summary.csv")
    save_csv(pd.DataFrame(tuned_per_class_rows), "tuned_holdout_per_class.csv")
    save_csv(pd.DataFrame(tuned_confusion_rows), "tuned_holdout_confusion_matrices.csv")
    save_csv(pd.DataFrame(best_tuned_weakest), "best_model_weakest_classes.csv")
    save_csv(pd.DataFrame(best_tuned_confusions), "best_model_top_confusions.csv")
    save_csv(pd.DataFrame(latency_rows), "baseline_latency.csv")

    json_path = RESULTS_DIR / "experiment_results.json"
    with json_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_summary(result)
    return result


if __name__ == "__main__":
    completed = run_experiments()
    print("Completed validation summary")
    for row in completed["repeated_cv"]["summary"]:
        print(
            f"CV {row['model']}: "
            f"accuracy {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}; "
            f"macro F1 {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}; "
            f"weighted F1 {row['weighted_f1_mean']:.4f} ± "
            f"{row['weighted_f1_std']:.4f}; macro-F1 fold wins "
            f"{row['macro_f1_fold_wins']}/50."
        )
    for row in completed["tuned_holdout_evaluation"]["per_model_summary"]:
        print(
            f"Tuned holdout {row['model']}: best params={row['best_params']}; "
            f"accuracy={row['accuracy']:.4f}; macro F1={row['macro_f1']:.4f}; "
            f"weighted F1={row['weighted_f1']:.4f}."
        )
    for row in completed["latency"]["rows"]:
        print(
            f"Latency {row['model']}: median={row['median_ms']:.4f} ms, "
            f"P95={row['p95_ms']:.4f} ms, scaler included="
            f"{row['scaler_included']}."
        )
    print(f"Best tuned model by training CV macro F1: "
          f"{completed['tuned_holdout_evaluation']['best_model_by_training_cv_macro_f1']}")
    hardware = completed["environment"]["hardware"]
    print(
        f"Measured host: CPU={hardware['cpu_model'] or 'unavailable'}; "
        f"RAM_GiB={hardware['installed_ram_gib'] or 'unavailable'}; "
        f"OS={completed['environment']['platform']}."
    )
    print(f"Results directory: {RESULTS_DIR / 'results'}")
