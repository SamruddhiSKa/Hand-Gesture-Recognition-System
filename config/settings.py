import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "alphabets")
LANDMARK_CSV_PATH = os.path.join(BASE_DIR, "landmark_dataset.csv")

# MediaPipe Hand Detection
MEDIAPIPE_MAX_HANDS = 2
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
GESTURE_CONFIDENCE_THRESHOLD = 0.75
PREDICTION_SMOOTHING_WINDOW = 5
HOLD_REQUIRED_SECONDS = 3.0
COOLDOWN_REQUIRED_SECONDS = 1.0
NO_HAND_TIMEOUT_SECONDS = 8.0

# Model Features
NUM_LANDMARKS = 21         
NUM_COORDS_PER_LANDMARK = 2 
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS_PER_LANDMARK 

# App Info
APP_TITLE = "Sign Language to Text Converter"
APP_DESCRIPTION = "Real-time hand gesture recognition using MediaPipe and Machine Learning"
APP_ICON = "🖐️"

# WebRTC (for streamlit-webrtc)
WEBRTC_STUN_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
    {"urls": ["stun:stun2.l.google.com:19302"]},
]

WEBRTC_TURN_SECRET_KEYS = (
    "TURN_SERVER_URL",
    "TURN_SERVER_USERNAME",
    "TURN_SERVER_CREDENTIAL",
)

# UI Colors (BGR for CV2)
PREDICTION_TEXT_COLOR = (0, 255, 0)
PREDICTION_BG_COLOR = (0, 0, 0)
