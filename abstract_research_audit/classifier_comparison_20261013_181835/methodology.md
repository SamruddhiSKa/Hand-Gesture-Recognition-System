# Controlled Classifier Comparison

## Scope

Compare a newly fitted in-memory Random Forest, RBF SVM, and KNN using the
existing normalized 42-coordinate rows from the project landmark CSV. This is
not a normalization ablation: raw landmark coordinates cannot be recovered
from this CSV.

## Dataset and split

- Input: `../../landmark_dataset.csv`
- Expected dataset SHA-256: `29c682b9bd6d2fa63417edcb745ad5df6a188fc3e84e9a954a1eb43900ee1ca9`
- Expected rows/features/classes: 3,263 / 42 / 36
- Split: stratified 80:20, `random_state=42`
- Expected sizes: 2,610 training rows and 653 test rows
- One set of train/test indices is created before fitting and reused by all
  classifiers. The script checks all dataset and split assumptions before
  constructing or fitting estimators.
- No tuning or test-driven selection is performed.

## Fixed estimators

- Random Forest: `RandomForestClassifier(n_estimators=100, random_state=42)`.
- RBF SVM: training-fitted `StandardScaler` followed by
  `SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=None,
  probability=False, random_state=42)`.
- KNN: training-fitted `StandardScaler` followed by
  `KNeighborsClassifier(n_neighbors=5, weights="uniform",
  metric="minkowski", p=2)`, equivalent to Euclidean distance.

The scaler is inside each pipeline and is fitted only on training rows.
The persisted project model is not loaded, modified, or overwritten. The new
Random Forest is fitted in memory solely as a controlled comparator.

## Outputs

`classifier_comparison.json` contains the dataset SHA-256, class counts,
split settings and index hashes, software versions, estimator settings,
accuracy, macro F1, weighted F1, per-class precision/recall/F1/support,
fixed-order confusion matrices, classifier-only latency summaries, and
captured warnings or estimator failures.

Latency measures a warm one-row `predict()` call repeated over held-out rows
five times. Feature-row slicing is outside the timer. For SVM/KNN, the fitted
scaler transform is part of the estimator pipeline prediction and is included.
Camera capture, MediaPipe inference, frame conversion, rendering, and UI
latency are excluded.

## Separate preprocessing issue

Training extraction processes 200x200 image crops, while live inference passes
the video frame dimensions into the existing aspect-ratio correction. A
non-square live frame may therefore use a different x-coordinate correction
than the square training crops. This comparison intentionally leaves both
the existing dataset and live feature pipeline unchanged; it does not assess
or fix that discrepancy.
