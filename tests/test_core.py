import numpy as np
from types import SimpleNamespace

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
    }


def test_text_state_operations():
    assert add_letter("H", "I") == "HI"
    assert delete_letter("HI") == "H"
    assert complete_word("HI", ["BYE"]) == ("", ["BYE", "HI"])
    assert clear_text() == ("", [])