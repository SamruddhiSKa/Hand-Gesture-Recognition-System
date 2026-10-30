# Research validation run

## Design and integrity checks

- Dataset SHA-256: `29c682b9bd6d2fa63417edcb745ad5df6a188fc3e84e9a954a1eb43900ee1ca9`
- Rows/features/classes: 3263 / 42 / 36
- Holdout split: stratified 80:20, seed 42; 2610 train / 653 test.
- Repeated CV runs only on the 80% training partition so the fixed holdout remains outside CV and all grid-search model selection.
- No estimator or scaler is fitted on the test partition.
- No existing serialized model is loaded or saved.

## Repeated stratified CV on training partition

| Estimator | Accuracy (mean ± SD) | Macro F1 (mean ± SD) | Weighted F1 (mean ± SD) | Fold wins by macro F1 |
|---|---:|---:|---:|---:|
| random_forest | 0.9417 ± 0.0101 | 0.9327 ± 0.0120 | 0.9416 ± 0.0101 | 49 |
| knn | 0.9192 ± 0.0107 | 0.9072 ± 0.0128 | 0.9194 ± 0.0106 | 1 |
| rbf_svm | 0.8619 ± 0.0124 | 0.8354 ± 0.0183 | 0.8544 ± 0.0139 | 0 |

Fold-wise wins compare scores only within the same held-out fold. They do not establish signer/session-independent generalization.

## Grid search and one-time holdout evaluation

| Estimator | Best training-CV macro F1 | Best parameters | Holdout accuracy | Holdout macro F1 | Holdout weighted F1 |
|---|---:|---|---:|---:|---:|
| random_forest | 0.9323 | `{"classifier__max_depth": 20, "classifier__n_estimators": 300}` | 0.9617 | 0.9545 | 0.9613 |
| knn | 0.9315 | `{"classifier__n_neighbors": 3, "classifier__weights": "distance"}` | 0.9602 | 0.9527 | 0.9606 |
| rbf_svm | 0.9425 | `{"classifier__C": 100, "classifier__gamma": "scale"}` | 0.9525 | 0.9424 | 0.9522 |

Best configured estimator selected by training-only grid-search CV macro F1: **rbf_svm**. The fixed test set did not select the model.

## Baseline estimator latency

| Estimator | Median ms | P95 ms | Timed calls | Scaling included? |
|---|---:|---:|---:|---|
| random_forest | 2.8189 | 15.2866 | 3265 | no |
| knn | 1.5798 | 2.4531 | 3265 | yes, in pipeline |
| rbf_svm | 0.7538 | 1.9082 | 3265 | yes, in pipeline |

Timing is warm single-row `predict()` latency: row slicing excluded; scaler transforms included for KNN/SVM; camera, MediaPipe, rendering, and UI latency excluded.

## Unrun questions / limitations

- **normalization_ablation:** Not run: the available CSV stores only wrist-relative, max-absolute-scale-normalized features. Raw coordinates and the pre-scale wrist-relative vectors cannot be recovered by inversion. The extraction code accepts image dimensions for aspect correction, but those original per-sample dimensions are not in the CSV. No source images were processed.
- **temporal_smoothing:** Not run: no labeled video/ordered sequence artifact was found in the pre-execution read-only audit. The available landmark table contains feature columns and label only, with no sequence/frame order or timestamps suitable for sequence-level evaluation.
- **signer_session_generalization:** Not run: the available landmark table contains no signer or session identifier, and no valid group-held-out split can be formed.

The training/live aspect-correction discrepancy remains separate and unmodified: extraction operates on square still crops and passes `image.shape`, while live inference passes full-frame dimensions to the aspect-correction function. This run does not test or correct it.

## Warnings and failures

- CV: 0 warning(s).
- Grid search/model evaluation: 0 warning(s).
- Latency fits: 0 warning(s).
- Per-model errors and warning details are recorded in `experiment_results.json`.
