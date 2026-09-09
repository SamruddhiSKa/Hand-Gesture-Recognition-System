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
    MEDIAPIPE_MAX_HANDS,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    PREDICTION_TEXT_COLOR,
    PREDICTION_BG_COLOR,
)

logger = logging.getLogger(__name__)

def create_hands_detector():
    """Create a MediaPipe Hands instance."""
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MEDIAPIPE_MAX_HANDS,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )

def extract_landmarks(hand_landmarks):
    """Normalized landmarks extraction (exactly like original code)."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y])
    
    coords = np.array(coords)
    coords = coords - coords[0] # subtract wrist
    scale = np.max(np.abs(coords))
    
    if scale != 0:
        coords = coords / scale
        
    return coords.flatten().reshape(1, -1)

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
    result = {"prediction": "", "confidence": 0.0, "hand_detected": False}

    try:
        results = detector.process(rgb)
    except Exception as e:
        logger.exception("MediaPipe frame processing error: %s", e)
        return frame, result

    if results.multi_hand_landmarks:
        result["hand_detected"] = True
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            
            # Predict
            features = extract_landmarks(hand_landmarks)
            if model is not None:
                try:
                    prediction = model.predict(features)[0]
                    result["prediction"] = str(prediction)
                    probas = model.predict_proba(features)[0]
                    result["confidence"] = round(float(np.max(probas)) * 100, 1)
                except AttributeError:
                    result["confidence"] = 0.0
                except Exception as e:
                    logger.exception("Gesture prediction error: %s", e)

    # Draw result on frame
    if result["prediction"]:
        label = f"{result['prediction']}"
        cv2.putText(frame, label, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, PREDICTION_TEXT_COLOR, 2)

    return frame, result
