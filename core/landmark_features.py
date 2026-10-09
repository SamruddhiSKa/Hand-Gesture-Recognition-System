"""Shared MediaPipe landmark-to-feature transformation."""

import numpy as np


def extract_landmarks(hand_landmarks, image_shape=None):
    """Return wrist-relative, max-absolute-scaled x/y features as one row."""
    landmarks = getattr(hand_landmarks, "landmark", None)
    if landmarks is None or len(landmarks) != 21:
        raise ValueError("Expected exactly 21 hand landmarks")

    coords = np.asarray(
        [[landmark.x, landmark.y] for landmark in landmarks],
        dtype=np.float64,
    )
    if coords.shape != (21, 2) or not np.isfinite(coords).all():
        raise ValueError("Landmarks must contain finite x/y coordinates")

    if image_shape is not None:
        height, width = image_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Image dimensions must be positive")
        coords[:, 0] *= width / height

    coords -= coords[0]
    scale = np.max(np.abs(coords))
    if scale != 0:
        coords /= scale

    return coords.flatten().reshape(1, -1)