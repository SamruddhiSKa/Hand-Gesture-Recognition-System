"""Reproducible offline benchmark for the trained landmark classifier.

This benchmark measures held-out landmark classification and warm model inference.
It does not measure webcam capture, MediaPipe detection, WebRTC, or deployment load.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "landmark_dataset.csv"
DEFAULT_MODEL = ROOT_DIR / "models" / "gesture_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CONFIDENCE_THRESHOLD = 0.75


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values), value))


def measure_predictions(model, features: pd.DataFrame, repeats: int) -> tuple[np.ndarray, list[float]]:
    """Return predictions and per-row warm inference times in milliseconds."""
    model.predict(features.iloc[:1])
    timings: list[float] = []
    predictions = None
    for _ in range(repeats):
        for row_index in range(len(features)):
            row = features.iloc[[row_index]]
            started = time.perf_counter_ns()
            current_prediction = model.predict(row)[0]
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            timings.append(elapsed_ms)
            if predictions is None:
                predictions = np.empty(len(features), dtype=object)
            predictions[row_index] = current_prediction
    assert predictions is not None
    return predictions, timings


def build_report(result: dict) -> str:
    performance = result["performance"]
    quality = result["quality"]
    reliability = result["reliability"]
    lines = [
        "========================================",
        "PROJECT BENCHMARK REPORT",
        "========================================",
        f"PROJECT: {result['project']}",
        f"PURPOSE: {result['purpose']}",
        f"TECHNOLOGY: {result['technology']}",
        "ARCHITECTURE: Streamlit UI -> streamlit-webrtc callback -> MediaPipe Hands -> normalized landmarks -> Random Forest classifier.",
        "",
        "----------------------------------------",
        "DATA / TEST SET",
        "----------------------------------------",
        f"Dataset rows: {result['dataset']['rows']}",
        f"Feature count: {result['dataset']['features']}",
        f"Classes: {result['dataset']['classes']}",
        f"Held-out test rows: {result['dataset']['test_rows']}",
        f"Split: stratified {TEST_SIZE:.0%}, random_state={RANDOM_STATE}",
        "Input: repository landmark vectors; no synthetic data used.",
        "",
        "----------------------------------------",
        "PERFORMANCE (WARM MODEL INFERENCE)",
        "----------------------------------------",
        f"Mean: {performance['mean_ms']:.4f} ms/row",
        f"Median: {performance['median_ms']:.4f} ms/row",
        f"P95: {performance['p95_ms']:.4f} ms/row",
        f"P99: {performance['p99_ms']:.4f} ms/row",
        f"Minimum: {performance['min_ms']:.4f} ms/row",
        f"Maximum: {performance['max_ms']:.4f} ms/row",
        f"Throughput: {performance['rows_per_second']:.2f} rows/second ({performance['rows_per_minute']:.2f} rows/minute)",
        f"Timing samples: {performance['timing_samples']}",
        "",
        "----------------------------------------",
        "RELIABILITY / QUALITY",
        "----------------------------------------",
        f"Total predictions: {reliability['total']}",
        f"Correct: {reliability['successful']}",
        f"Incorrect: {reliability['failed']}",
        f"Success rate (exact classification): {reliability['success_rate_percent']:.4f}%",
        f"Error rate (exact classification): {reliability['error_rate_percent']:.4f}%",
        f"Accuracy: {quality['accuracy_percent']:.4f}%",
        f"Macro F1: {quality['macro_f1']:.6f}",
        f"Weighted F1: {quality['weighted_f1']:.6f}",
        f"Confidence threshold: {quality['confidence_threshold']:.0%}",
        f"Accepted prediction coverage: {quality['accepted_coverage_percent']:.4f}%",
        f"Accepted prediction accuracy: {quality['accepted_accuracy_percent']:.4f}%",
        f"Majority baseline accuracy: {result['baseline']['accuracy_percent']:.4f}%",
        f"Absolute accuracy lift vs majority baseline: {result['baseline']['absolute_accuracy_lift_percentage_points']:.4f} percentage points",
        "Per-class metrics and confusion matrix are included in benchmark_results.json.",
        "",
        "----------------------------------------",
        "RESOURCE / SCOPE",
        "----------------------------------------",
        f"Model file: {result['resources']['model_path']} ({result['resources']['model_size_bytes']} bytes)",
        f"CPU/GPU/network: not measured by this offline benchmark",
        "Concurrency/load testing: not applicable to this local model script; Streamlit/WebRTC deployment was not exercised.",
        "",
        "----------------------------------------",
        "LIMITATIONS",
        "----------------------------------------",
        "Held-out rows come from the repository landmark CSV and may share capture conditions or subjects with training data.",
        "The benchmark excludes camera quality, hand detection failures, landmark extraction time, browser/WebRTC overhead, and network latency.",
        "Results are hardware-, Python-, dependency-, and model-file-version-specific; they are not production guarantees.",
    ]
    return "\n".join(lines)


def run_benchmark(dataset_path: Path, model_path: Path, repeats: int) -> dict:
    data = pd.read_csv(dataset_path)
    features = data.drop(columns=["label"])
    labels = data["label"]
    _, test_features, _, test_labels = train_test_split(
        features, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )

    model = joblib.load(model_path)
    predictions, timings = measure_predictions(model, test_features, repeats)
    correct = int(np.sum(predictions == test_labels.to_numpy()))
    total = len(test_labels)
    report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
    probabilities = model.predict_proba(test_features).max(axis=1)
    accepted = probabilities >= CONFIDENCE_THRESHOLD
    accepted_accuracy = float(accuracy_score(test_labels.to_numpy()[accepted], predictions[accepted]))

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(features, labels)
    baseline_accuracy = float(accuracy_score(test_labels, majority.predict(test_features)))

    mean_ms = float(np.mean(timings))
    return {
        "project": "Hand Gesture Recognition System",
        "purpose": "Recognize alphabet and digit hand gestures from webcam hand landmarks and build text.",
        "technology": "Python, Streamlit, streamlit-webrtc, MediaPipe Hands, OpenCV, NumPy, pandas, scikit-learn, joblib",
        "benchmark_scope": "Offline warm inference on precomputed landmark vectors",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "dataset": {
            "path": str(dataset_path),
            "rows": int(len(data)),
            "features": int(features.shape[1]),
            "classes": int(labels.nunique()),
            "class_counts": {str(key): int(value) for key, value in labels.value_counts().sort_index().items()},
            "test_rows": int(total),
            "test_class_counts": {str(key): int(value) for key, value in test_labels.value_counts().sort_index().items()},
        },
        "performance": {
            "mean_ms": mean_ms,
            "median_ms": percentile(timings, 50),
            "p95_ms": percentile(timings, 95),
            "p99_ms": percentile(timings, 99),
            "min_ms": float(np.min(timings)),
            "max_ms": float(np.max(timings)),
            "rows_per_second": 1000 / mean_ms,
            "rows_per_minute": 60_000 / mean_ms,
            "timing_samples": len(timings),
            "repeats": repeats,
        },
        "reliability": {
            "total": total,
            "successful": correct,
            "failed": total - correct,
            "success_rate_percent": correct / total * 100,
            "error_rate_percent": (total - correct) / total * 100,
        },
        "quality": {
            "accuracy_percent": report["accuracy"] * 100,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "accepted_coverage_percent": float(np.mean(accepted) * 100),
            "accepted_accuracy_percent": accepted_accuracy * 100,
            "per_class": {
                str(key): value for key, value in report.items() if key not in {"accuracy", "macro avg", "weighted avg"}
            },
            "confusion_matrix_labels": [str(value) for value in sorted(labels.unique())],
            "confusion_matrix": confusion_matrix(test_labels, predictions, labels=sorted(labels.unique())).tolist(),
        },
        "baseline": {
            "type": "majority-class DummyClassifier",
            "accuracy_percent": baseline_accuracy * 100,
            "absolute_accuracy_lift_percentage_points": (report["accuracy"] - baseline_accuracy) * 100,
        },
        "resources": {
            "model_path": str(model_path),
            "model_size_bytes": model_path.stat().st_size,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=ROOT_DIR / "benchmark_results.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    result = run_benchmark(args.dataset, args.model, args.repeats)
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(build_report(result))
    print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
