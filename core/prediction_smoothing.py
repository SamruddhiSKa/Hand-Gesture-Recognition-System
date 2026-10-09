"""Small utilities for stable per-frame gesture predictions."""

from collections import Counter


def stable_prediction(history: list[tuple[str, float]], min_votes: int) -> tuple[str, float]:
    """Return a label only after enough recent votes, with matching-label confidence."""
    if not history:
        return "", 0.0

    label, vote_count = Counter(item[0] for item in history).most_common(1)[0]
    if vote_count < min_votes:
        return "", 0.0

    matching_confidences = [confidence for candidate, confidence in history if candidate == label]
    mean_confidence = sum(matching_confidences) / len(matching_confidences)
    return label, round(mean_confidence, 1)