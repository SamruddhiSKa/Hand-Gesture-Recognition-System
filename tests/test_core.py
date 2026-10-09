import numpy as np
import joblib
from types import SimpleNamespace
import core.hand_processor as hand_processor

from core.hand_processor import extract_landmarks, process_frame
from core.model_manager import model_contract_error
from core.prediction_smoothing import stable_prediction
from core.text_state import add_letter, clear_text, complete_word, delete_letter
from config.settings import MODEL_PATH


def test_extract_landmarks_translates_and_scales_wrist():
    landmarks = SimpleNamespace(
        landmark=[SimpleNamespace(x=1.0, y=1.0), SimpleNamespace(x=3.0, y=5.0)]
        + [SimpleNamespace(x=1.0, y=1.0) for _ in range(19)]
    )

    result = extract_landmarks(landmarks)

    assert result.shape == (1, 42)
    np.testing.assert_allclose(result[0, :4], [0.0, 0.0, 0.5, 1.0])


def test_extract_landmarks_matches_square_training_coordinates():
    landmarks = SimpleNamespace(
        landmark=[SimpleNamespace(x=0.0, y=0.0), SimpleNamespace(x=1.0, y=2.0)]
        + [SimpleNamespace(x=0.0, y=0.0) for _ in range(19)]
    )

    result = extract_landmarks(landmarks, image_shape=(480, 640, 3))

    assert result.shape == (1, 42)
    np.testing.assert_allclose(result[0, :4], [0.0, 0.0, 2 / 3, 1.0])


def test_square_training_and_live_feature_paths_are_identical():
    landmarks = SimpleNamespace(
        landmark=[SimpleNamespace(x=0.1, y=0.2), SimpleNamespace(x=0.7, y=0.8)]
        + [SimpleNamespace(x=0.1, y=0.2) for _ in range(19)]
    )

    training_features = extract_landmarks(landmarks)
    live_features = extract_landmarks(landmarks, image_shape=(400, 400, 3))

    np.testing.assert_array_equal(training_features, live_features)


def test_extract_landmarks_rejects_missing_wrong_count_and_nonfinite_values():
    with np.testing.assert_raises(ValueError):
        extract_landmarks(None)
    with np.testing.assert_raises(ValueError):
        extract_landmarks(SimpleNamespace(landmark=[SimpleNamespace(x=0.0, y=0.0)]))

    bad_landmarks = SimpleNamespace(
        landmark=[SimpleNamespace(x=np.nan, y=0.0)]
        + [SimpleNamespace(x=0.0, y=0.0) for _ in range(20)]
    )
    with np.testing.assert_raises(ValueError):
        extract_landmarks(bad_landmarks)


def test_process_frame_returns_safe_no_hand_result():
    detector = SimpleNamespace(
        process=lambda image: SimpleNamespace(multi_hand_landmarks=[])
    )

    _, result = process_frame(np.zeros((8, 8, 3), dtype=np.uint8), detector, None)

    assert result == {
        "prediction": "",
        "candidate": "",
        "confidence": 0.0,
        "hand_detected": False,
        "multiple_hands": False,
        "landmark_count": 0,
        "feature_shape": None,
    }


def test_text_state_operations():
    assert add_letter("H", "I") == "HI"
    assert delete_letter("HI") == "H"
    assert complete_word("HI", ["BYE"]) == ("", ["BYE", "HI"])
    assert clear_text() == ("", [])


def test_process_frame_preserves_model_prediction_contract(monkeypatch):
    landmarks = SimpleNamespace(
        landmark=[SimpleNamespace(x=0.0, y=0.0)]
        + [SimpleNamespace(x=float(index) / 20, y=float(index) / 20) for index in range(1, 21)]
    )
    detector = SimpleNamespace(
        process=lambda image: SimpleNamespace(multi_hand_landmarks=[landmarks])
    )
    monkeypatch.setattr(hand_processor.mp_draw, "draw_landmarks", lambda *args, **kwargs: None)

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    low_confidence_model = SimpleNamespace(
        n_features_in_=42,
        classes_=np.array(["a", "b"]),
        predict_proba=lambda features: np.array([[0.5, 0.5]]),
    )
    _, rejected = hand_processor.process_frame(frame.copy(), detector, low_confidence_model, flip=False)
    assert rejected["candidate"] in {"a", "b"}
    assert rejected["confidence"] == 50.0
    assert rejected["prediction"] == ""

    confident_model = SimpleNamespace(
        n_features_in_=42,
        classes_=np.array(["a", "b"]),
        predict_proba=lambda features: np.array([[0.9, 0.1]]),
    )
    _, accepted = hand_processor.process_frame(frame.copy(), detector, confident_model, flip=False)

    assert accepted["landmark_count"] == 21
    assert accepted["feature_shape"] == [1, 42]
    assert accepted["candidate"] == accepted["prediction"] == "a"
    assert accepted["confidence"] == 90.0


def test_production_model_matches_expected_contract():
    model = joblib.load(MODEL_PATH)

    assert model_contract_error(model) is None


def test_model_contract_rejects_wrong_feature_count():
    invalid_model = SimpleNamespace(
        n_features_in_=41,
        classes_=tuple("0123456789abcdefghijklmnopqrstuvwxyz"),
        predict_proba=lambda features: np.ones((len(features), 36)) / 36,
    )

    assert model_contract_error(invalid_model) == "Expected 42 input features"


def test_temporal_vote_requires_three_and_matches_confidence_to_winner():
    history = [("h", 90.0), ("g", 99.0), ("h", 80.0), ("h", 70.0), ("g", 99.0)]

    assert stable_prediction(history[:2], min_votes=3) == ("", 0.0)
    assert stable_prediction(history, min_votes=3) == ("h", 80.0)