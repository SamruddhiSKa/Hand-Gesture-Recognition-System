# Project Benchmark Report

## Project

**Project:** Hand Gesture Recognition System

**Purpose:** Recognize alphabet and digit hand gestures from webcam hand landmarks and build text.

**Technology:** Python, Streamlit, streamlit-webrtc, MediaPipe Hands, OpenCV, NumPy, pandas, scikit-learn, and joblib.

**Architecture:** Streamlit UI -> streamlit-webrtc callback -> MediaPipe Hands -> normalized 21-point hand landmarks -> persisted Random Forest classifier -> prediction and word builder.

## Data / Test Set

Measured from `landmark_dataset.csv`:

- 3,263 rows
- 42 numeric features per row: 21 landmarks x/y coordinates
- 36 classes: digits `0`-`9` and letters `a`-`z`
- 653 held-out test rows
- Stratified 80/20 split with `random_state=42`
- No synthetic data used

The split is reproducible, but the CSV does not document subject or capture-session IDs. Therefore, this is a row-level held-out evaluation, not a subject-independent generalization test.

## Performance

Warm `RandomForestClassifier.predict()` was measured one row at a time on the 653 held-out landmark rows, after one warm-up prediction. Five repetitions produced 3,265 timing samples:

| Metric | Measured result |
| --- | ---: |
- Mean latency | 3.3954 ms/row |
- Median latency | 3.0650 ms/row |
- P95 latency | 5.0407 ms/row |
- P99 latency | 6.8487 ms/row |
- Minimum latency | 2.1076 ms/row |
- Maximum latency | 41.1702 ms/row |
- Throughput derived from mean | 294.52 rows/second |
- Throughput derived from mean | 17,671.03 rows/minute |

These are classifier-only figures. They are not end-to-end frame-processing FPS.

## Reliability / ML Quality

On the held-out rows:

- Total predictions: 653
- Correct: 626
- Incorrect: 27
- Exact-classification success rate: 95.8652%
- Exact-classification error rate: 4.1348%
- Accuracy: 95.8652%
- Macro F1: 0.951759
- Weighted F1: 0.958116
- Confidence rejection threshold: 75%
- Accepted prediction coverage: 75.65%
- Accuracy among accepted predictions: 100.0000%

Per-class precision, recall, F1, support, and the confusion matrix are in `benchmark_results.json`.

## Baseline Comparison

The legitimate baseline is a `DummyClassifier(strategy="most_frequent")` that always predicts the most common training label under the same held-out split:

- Baseline accuracy: 4.5942%
- Current model accuracy: 95.8652%
- Absolute accuracy lift: 91.2711 percentage points

This is a sanity-check baseline, not evidence of an optimization improvement over a previous implementation. No previous version was found in the repository.

## Resource Usage and Scope

- Current model file size: 16,375,897 bytes
- CPU utilization: not measured
- Memory usage: not measured
- GPU usage: not measured
- Disk/network usage during inference: not measured
- Concurrent requests/users: not tested
- Browser/WebRTC load test: not performed
- MediaPipe detection and landmark extraction latency: not measured

## Strongest Resume-Suitable Metrics

1. **95.8652% held-out accuracy across 653 test landmark samples and 36 classes.** Measured by comparing model predictions with labels from a deterministic stratified 20% split.
2. **0.951759 macro F1 across 36 classes.** Calculated with scikit-learn from the same held-out predictions, giving each class equal weight.
3. **3.0650 ms median warm classifier inference latency per landmark row.** Measured with `time.perf_counter_ns()` over 3,265 per-row prediction timings after warm-up.
4. **294.52 landmark predictions/second derived from mean latency.** Calculated as `1000 / mean_latency_ms`; this is local classifier throughput, not application FPS.
5. **75.65% accepted-prediction coverage at a 75% confidence threshold, with 100.0000% accuracy on accepted held-out rows.** The remaining rows are presented as unclear rather than forced into a class.

## Resume Bullets

- Engineered a Python, MediaPipe, and scikit-learn hand-gesture recognition system covering 36 alphabet and digit classes, achieving 95.8652% accuracy across 653 held-out landmark samples.
- Developed a Random Forest landmark classifier with 0.951759 macro F1 on a deterministic stratified evaluation split, supporting per-class gesture quality analysis.
- Integrated warm landmark inference into a Streamlit/WebRTC application, measuring 3.0650 ms median classifier latency per input row and 294.52 derived predictions per second locally.

## Methodology

`python tools/benchmark.py --repeats 5` loads the CSV and persisted `models/gesture_model.pkl`, creates the same deterministic stratified split used by training, warms the model once, times each held-out row individually for five repetitions, and calculates accuracy, F1, per-class metrics, confusion matrix, latency percentiles, and model size. Raw structured output is written to `benchmark_results.json`.

## Limitations

- Held-out rows may share subjects, backgrounds, devices, or capture sessions with training data because the dataset has no documented group identifiers.
- The benchmark uses precomputed landmarks, so it excludes camera quality, MediaPipe hand-detection failures, landmark extraction cost, frame annotation, browser/WebRTC overhead, and network latency.
- The timing environment was Windows 10 with Python 3.11.8, NumPy 1.26.4, pandas 2.2.2, and scikit-learn 1.5.2; results are hardware- and environment-specific.
- Five repetitions and 3,265 timing samples characterize this local run; latency varied between benchmark runs and does not establish a production service SLO.
- No concurrency or stress test was run, so no concurrent-user or production-capacity claim is justified.