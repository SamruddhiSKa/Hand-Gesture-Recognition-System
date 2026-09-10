import numpy as np
from types import SimpleNamespace
from sklearn.ensemble import RandomForestClassifier
import core.hand_processor as hand_processor

from core.hand_processor import extract_landmarks, process_frame
from core.text_state import add_letter, clear_text, complete_word, delete_letter


def test_extract_landmarks_translates_and_scales_wrist():
    landmarks = SimpleNamespace(
        landmark=[SimpleNamespace(x=1.0, y=1.0), SimpleNamespace(x=3.0, y=5.0)]
    )

    result = extract_landmarks(landmarks)

    np.testing.assert_allclose(result, [[0.0, 0.0, 0.5, 1.0]])


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
    model = RandomForestClassifier(n_estimators=1, random_state=42)
    model.fit(np.zeros((2, 42)), ["a", "b"])
    monkeypatch.setattr(hand_processor.mp_draw, "draw_landmarks", lambda *args, **kwargs: None)

    _, result = hand_processor.process_frame(
        np.zeros((32, 32, 3), dtype=np.uint8), detector, model, flip=False
    )

    assert result["landmark_count"] == 21
    assert result["feature_shape"] == [1, 42]
    assert result["candidate"] in {"a", "b"}