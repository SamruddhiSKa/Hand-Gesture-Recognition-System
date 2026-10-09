import cv2
import numpy as np
import mediapipe as mp
import logging
import streamlit as st

try:
    from mediapipe.solutions import hands as mp_hands
    from mediapipe.solutions import drawing_utils as mp_draw
    from mediapipe.solutions import drawing_styles as mp_styles
except:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_draw
    import mediapipe.python.solutions.drawing_styles as mp_styles

from config.settings import (
    GESTURE_CONFIDENCE_THRESHOLD,
    MEDIAPIPE_MAX_HANDS,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    PREDICTION_TEXT_COLOR,
    PREDICTION_BG_COLOR,
)
from core.landmark_features import extract_landmarks

logger = logging.getLogger(__name__)
_feature_shape_logged = False

def create_hands_detector():
    """Create a MediaPipe Hands instance."""
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MEDIAPIPE_MAX_HANDS,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )

def process_frame(frame, detector, model, flip=True):
    """Processes frame to return (annotated_frame, result_dict).
    
    result_dict contains:
        - prediction: str (the predicted letter/digit)
        - confidence: float (0-100, prediction confidence %)
        - hand_detected: bool
    """
    if flip:
        frame = cv2.flip(frame, 1) # horizontal flip for selfie view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = {
        "prediction": "",
        "candidate": "",
        "confidence": 0.0,
        "hand_detected": False,
        "multiple_hands": False,
        "landmark_count": 0,
        "feature_shape": None,
    }

    try:
        results = detector.process(rgb)
    except Exception as e:
        logger.exception("MediaPipe frame processing error: %s", e)
        return frame, result

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) > 1:
            result["multiple_hands"] = True
            return frame, result
        result["hand_detected"] = True
        for hand_landmarks in results.multi_hand_landmarks:
            try:
                result["landmark_count"] = len(hand_landmarks.landmark)
            except (AttributeError, TypeError):
                logger.warning("Invalid MediaPipe hand landmarks")
                continue
            if result["landmark_count"] != 21:
                logger.warning("Unexpected landmark count: %s", result["landmark_count"])
                continue
            # Draw landmarks
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            
            # Predict
            try:
                features = extract_landmarks(hand_landmarks, frame.shape)
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning("Invalid hand feature input: %s", e)
                continue
            result["feature_shape"] = list(features.shape)
            global _feature_shape_logged
            if not _feature_shape_logged:
                logger.info("Live gesture feature shape: %s", features.shape)
                _feature_shape_logged = True
            if model is not None:
                try:
                    expected_features = getattr(model, "n_features_in_", features.shape[1])
                    if expected_features != features.shape[1]:
                        raise ValueError(
                            f"Model expects {expected_features} features, received {features.shape[1]}"
                        )
                    probas = model.predict_proba(features)[0]
                    prediction = model.classes_[int(np.argmax(probas))]
                    result["candidate"] = str(prediction)
                    result["confidence"] = round(float(np.max(probas)) * 100, 1)
                except Exception as e:
                    logger.exception("Gesture prediction error: %s", e)
                    continue

                if result["confidence"] >= GESTURE_CONFIDENCE_THRESHOLD * 100:
                    result["prediction"] = result["candidate"]

    return frame, result
