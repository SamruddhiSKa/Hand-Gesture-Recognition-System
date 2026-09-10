import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import streamlit as st
import av
import time
import threading
import logging
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from core.model_manager import load_model
from core.hand_processor import create_hands_detector, process_frame
from core.text_state import add_letter, clear_text, complete_word, delete_letter
from config.settings import (
    WEBRTC_STUN_SERVERS,
    WEBRTC_TURN_SECRET_KEYS,
    APP_TITLE,
    APP_DESCRIPTION,
    APP_ICON,
)
from config.settings import (
    COOLDOWN_REQUIRED_SECONDS,
    HOLD_REQUIRED_SECONDS,
    NO_HAND_TIMEOUT_SECONDS,
    PREDICTION_SMOOTHING_WINDOW,
)

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# ─── Session State ──────────────────────────────────────────
if 'word' not in st.session_state:
    st.session_state.word = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = ""
if 'last_confidence' not in st.session_state:
    st.session_state.last_confidence = 0.0
if 'completed_words' not in st.session_state:
    st.session_state.completed_words = []
if 'camera_mode' not in st.session_state:
    st.session_state.camera_mode = "user" # Default to front camera

# ─── Auto-Capture Timers ──────────────────────────────────
if 'hold_start_time' not in st.session_state:
    st.session_state.hold_start_time = None
if 'is_cooling_down' not in st.session_state:
    st.session_state.is_cooling_down = False
if 'cooldown_start_time' not in st.session_state:
    st.session_state.cooldown_start_time = 0
if 'no_hand_start_time' not in st.session_state:
    st.session_state.no_hand_start_time = None
if 'last_added_letter' not in st.session_state:
    st.session_state.last_added_letter = ""
if 'hands_detector' not in st.session_state:
    st.session_state.hands_detector = None

# ─── Thread-safe Shared State ──────────────────────────────
class SharedState:
    """Thread-safe container to pass predictions from callback thread to main thread."""
    def __init__(self):
        self._result = {
            "prediction": "",
            "candidate": "",
            "confidence": 0.0,
            "hand_detected": False,
            "multiple_hands": False,
        }
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)
        self._flip = True
        self._frames_received = 0
        self._last_error = ""
        self._video_ended = False

    def mark_frame_received(self):
        with self._lock:
            self._frames_received += 1
            self._last_error = ""
            self._video_ended = False

    def mark_error(self, message):
        with self._lock:
            self._last_error = message

    def mark_video_ended(self):
        with self._lock:
            self._video_ended = True

    def get_diagnostics(self):
        with self._lock:
            return {
                "frames_received": self._frames_received,
                "last_error": self._last_error,
                "video_ended": self._video_ended,
            }

    def update(self, raw_result):
        with self._lock:
            if raw_result.get("prediction"):
                self._buffer.append(raw_result["prediction"])
            else:
                self._buffer.clear()
            if self._buffer:
                most_common = Counter(self._buffer).most_common(1)[0][0]
                self._result = {
                    "prediction": most_common,
                    "candidate": raw_result.get("candidate", ""),
                    "confidence": raw_result["confidence"],
                    "hand_detected": raw_result["hand_detected"],
                    "multiple_hands": raw_result.get("multiple_hands", False),
                }
            else:
                self._result = raw_result

    def set_flip(self, flip):
        with self._lock:
            self._flip = flip

    def get_flip(self):
        with self._lock:
            return self._flip

    def get(self):
        with self._lock:
            return self._result.copy()

if 'shared_state' not in st.session_state:
    st.session_state.shared_state = SharedState()

shared = st.session_state.shared_state


def get_ice_servers():
    """Build ICE servers without committing TURN credentials."""
    servers = list(WEBRTC_STUN_SERVERS)
    try:
        turn_url = st.secrets.get(WEBRTC_TURN_SECRET_KEYS[0])
        turn_username = st.secrets.get(WEBRTC_TURN_SECRET_KEYS[1])
        turn_credential = st.secrets.get(WEBRTC_TURN_SECRET_KEYS[2])
    except Exception:
        turn_url = turn_username = turn_credential = None

    if turn_url and turn_username and turn_credential:
        servers.append(
            {
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_credential,
            }
        )
    return servers

# ─── Custom CSS (Compacted) ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 1200px; }
    .stApp { background: #F5F7FA; }

    /* Compact Header */
    .app-header {
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 0.8rem;
    }
    .app-header h1 {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1B4F72;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header .subtitle {
        font-size: 0.8rem;
        color: #7F8C8D;
        margin: 0;
    }

    /* Compact Card */
    .card {
        background: #FFFFFF;
        border: 1px solid #E4EAF0;
        border-radius: 10px;
        padding: 0.8rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 0.5rem;
    }
    .card-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: #1B4F72;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #F0F3F7;
    }
    .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
    .dot-blue { background: #2E86C1; }
    .dot-green { background: #27AE60; }

    /* Compact Prediction */
    .prediction-letter {
        font-size: 4rem;
        font-weight: 900;
        color: #27AE60;
        text-align: center;
        margin: 0;
    }
    .prediction-label {
        font-size: 0.6rem;
        font-weight: 600;
        color: #95A5A6;
        text-transform: uppercase;
        text-align: center;
        margin-top: 0.2rem;
    }

    .word-display { text-align: center; }
    .word-label { font-size: 0.65rem; font-weight: 700; color: #7F8C8D; text-transform: uppercase; }
    .word-text { font-size: 1.4rem; font-weight: 800; color: #1B4F72; letter-spacing: 3px; }
    .word-text-empty { font-size: 1.4rem; font-weight: 200; color: #D5DBDB; }

    .status-box {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        padding: 0.4rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .status-detecting { background: #EAFAF1; color: #1E8449; }
    .status-no-hand { background: #FEF9E7; color: #9A7D0A; }
    .status-inactive { background: #F2F4F4; color: #85929E; }

    /* Compact Buttons */
    .stButton > button {
        font-size: 0.75rem !important;
        padding: 0.3rem 0.6rem !important;
        border-radius: 8px !important;
    }

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    video { border-radius: 8px; width: 100% !important; background: #000; }

    /* Progress Bar */
    .progress-container {
        width: 100%;
        background-color: #EBF5FB;
        border-radius: 10px;
        margin-top: 5px;
        height: 8px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #2E86C1, #27AE60);
        transition: width 0.1s linear;
    }
    .cooldown-status {
        font-size: 0.65rem;
        color: #E67E22;
        font-weight: 700;
        text-align: center;
        margin-top: 0.3rem;
        text-transform: uppercase;
    }
    .auto-status {
        font-size: 0.6rem;
        color: #7F8C8D;
        text-align: center;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <h1>{APP_TITLE}</h1>
    <p class="subtitle">Real-time Prediction</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Resources ─────────────────────────────────────────
model = load_model()
detector = st.session_state.hands_detector
if detector is None:
    try:
        detector = create_hands_detector()
        st.session_state.hands_detector = detector
    except Exception as e:
        logging.exception("MediaPipe initialization error: %s", e)
        st.error("Hand tracking could not be initialized. Please try again later.")

# ─── Video Callback ─────────────────────────────────────────
def video_frame_callback(frame):
    try:
        shared.mark_frame_received()
        img = frame.to_ndarray(format="bgr24")
        if detector is None or model is None:
            return frame
        # Get flip state from shared object
        should_flip = shared.get_flip()
        
        annotated_frame, result = process_frame(img, detector, model, flip=should_flip)
        shared.update(result)
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    except Exception as e:
        shared.mark_error("Frame processor error")
        logging.exception("Error in callback: %s", e)
        return frame


def on_video_ended():
    shared.mark_video_ended()
    logging.info("WebRTC video track ended")

# ─── Main Layout ────────────────────────────────────────────
col_feed, col_predict = st.columns([1.5, 1])

with col_feed:
    # Camera Toggle
    cam_col1, cam_col2 = st.columns([2, 1])
    with cam_col1:
        camera_choice = st.radio(
            "Select Camera:",
            options=["Front Camera", "Back Camera"],
            index=0 if st.session_state.camera_mode == "user" else 1,
            horizontal=True,
            key="cam_selector_radio"
        )
        st.session_state.camera_mode = "user" if camera_choice == "Front Camera" else "environment"
        shared.set_flip(st.session_state.camera_mode == "user")

    webrtc_ctx = webrtc_streamer(
        key="gesture-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "facingMode": st.session_state.camera_mode,
                "width": {"ideal": 640},
                "height": {"ideal": 480}
            },
            "audio": False
        },
        async_processing=True,
        on_video_ended=on_video_ended,
    )
    if webrtc_ctx.state.signalling:
        st.info("Connecting to camera...")
    elif not webrtc_ctx.state.playing:
        st.warning("Camera connection is not active. Click Start and allow camera access in your browser. If the connection failed, refresh the page and try again.")

with col_predict:
    # Prediction Results
    st.markdown("""
    <div class="card">
        <div class="card-title"><span class="dot dot-green"></span> Prediction</div>
    """, unsafe_allow_html=True)
    prediction_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # Word Builder (Merged for compactness)
    st.markdown("""
    <div class="card">
        <div class="card-title"><span class="dot dot-blue"></span> Word Builder</div>
    """, unsafe_allow_html=True)
    word_placeholder = st.empty()
    
    # Action Buttons (High placement)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        continue_btn = st.button("✚ Add Letter", use_container_width=True)
    with b_col2:
        end_btn = st.button("✔ End Word", use_container_width=True)
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        delete_btn = st.button("⌫ Delete Letter", use_container_width=True)
    with d_col2:
        clear_btn = st.button("✕ Clear All", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    completed_placeholder = st.empty()

# ─── Button Actions ─────────────────────────────────────────
if continue_btn and st.session_state.last_prediction:
    st.session_state.word = add_letter(st.session_state.word, st.session_state.last_prediction)

if end_btn and st.session_state.word:
    st.session_state.word, st.session_state.completed_words = complete_word(
        st.session_state.word, st.session_state.completed_words
    )

if delete_btn and st.session_state.word:
    st.session_state.word = delete_letter(st.session_state.word)

if clear_btn:
    st.session_state.word, st.session_state.completed_words = clear_text()

# ─── UI Updates ─────────────────────────────────────────────
if st.session_state.completed_words:
    sentence = " | ".join(st.session_state.completed_words)
    completed_placeholder.markdown(f"""
    <div class="card" style="margin-top: 5px; border-color: #27AE60;">
        <div class="word-label" style="color: #27AE60;">Sentences</div>
        <div style="font-size: 0.9rem; font-weight: 600;">{sentence}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Live Status Refresh ───────────────────────────────────
HOLD_REQUIRED = HOLD_REQUIRED_SECONDS
COOLDOWN_REQUIRED = COOLDOWN_REQUIRED_SECONDS
NO_HAND_TIMEOUT = NO_HAND_TIMEOUT_SECONDS


def render_live_status():
    """Render the current status without creating a separate fragment lifecycle."""
    now = time.time()
    result = shared.get()
    pred = result.get("prediction", "")
    candidate = result.get("candidate", "")
    hand_visible = result.get("hand_detected", False)
    multiple_hands = result.get("multiple_hands", False)
    diagnostics = shared.get_diagnostics()

    if webrtc_ctx.state.signalling:
        connection_status = "Connecting to camera..."
    elif webrtc_ctx.state.playing:
        connection_status = "Camera connected"
    elif diagnostics["last_error"] or diagnostics["video_ended"]:
        connection_status = "Camera connection failed - check camera permissions/network."
    else:
        connection_status = "Requesting camera access..."

    st.caption(
        f"WebRTC: {connection_status} | ICE: browser-managed STUN | "
        f"Processor: {'active' if diagnostics['frames_received'] else 'waiting'} | "
        f"Frames received: {diagnostics['frames_received']}"
    )

    if not webrtc_ctx.state.playing:
        prediction_placeholder.markdown(
            f"<div class='status-box status-inactive'>{connection_status}</div>",
            unsafe_allow_html=True,
        )
        word_placeholder.markdown(
            "<div class='word-display'><div class='word-text-empty'>-</div></div>",
            unsafe_allow_html=True,
        )
        return

    if st.session_state.is_cooling_down and now - st.session_state.cooldown_start_time >= COOLDOWN_REQUIRED:
        st.session_state.is_cooling_down = False

    if multiple_hands:
        st.session_state.hold_start_time = None
        st.session_state.last_prediction = ""
    elif hand_visible:
        st.session_state.no_hand_start_time = None
        if not st.session_state.is_cooling_down:
            if pred and pred == st.session_state.last_prediction:
                if st.session_state.hold_start_time is None:
                    st.session_state.hold_start_time = now
                hold_elapsed = now - st.session_state.hold_start_time
                if hold_elapsed >= HOLD_REQUIRED:
                    st.session_state.word = add_letter(st.session_state.word, pred)
                    st.session_state.is_cooling_down = True
                    st.session_state.cooldown_start_time = now
                    st.session_state.hold_start_time = None
            else:
                st.session_state.hold_start_time = now
        st.session_state.last_prediction = pred
    else:
        st.session_state.hold_start_time = None
        if st.session_state.no_hand_start_time is None:
            st.session_state.no_hand_start_time = now
        if now - st.session_state.no_hand_start_time >= NO_HAND_TIMEOUT and st.session_state.word:
            st.session_state.word, st.session_state.completed_words = complete_word(
                st.session_state.word, st.session_state.completed_words
            )
            st.session_state.no_hand_start_time = None

    if multiple_hands:
        prediction_placeholder.markdown(
            "<div class='prediction-letter' style='color: #E67E22;'>!</div><div class='status-box status-no-hand'>Use one hand</div>",
            unsafe_allow_html=True,
        )
    elif hand_visible and pred:
        hold_elapsed = (now - st.session_state.hold_start_time) if st.session_state.hold_start_time else 0
        pct = min(int((hold_elapsed / HOLD_REQUIRED) * 100), 100)
        prediction_placeholder.markdown(
            f"<div class='prediction-letter'>{pred}</div><div class='prediction-label'>Confidence {result.get('confidence', 0):.1f}%</div><div class='auto-status'>Hold {max(0, int(HOLD_REQUIRED - hold_elapsed))}s more...</div><div class='progress-container'><div class='progress-fill' style='width: {pct}%;'></div></div>",
            unsafe_allow_html=True,
        )
    elif hand_visible and candidate:
        prediction_placeholder.markdown(
            "<div class='prediction-letter' style='color: #E67E22;'>?</div><div class='status-box status-no-hand'>Gesture unclear</div><div class='auto-status'>Move your hand into a clearer pose</div>",
            unsafe_allow_html=True,
        )
    elif hand_visible:
        prediction_placeholder.markdown(
            "<div class='prediction-letter' style='color: #D5DBDB;'>?</div><div class='status-box status-no-hand'>Detecting gesture...</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.word:
        remaining = max(0, int(NO_HAND_TIMEOUT - (now - st.session_state.no_hand_start_time)))
        prediction_placeholder.markdown(
            f"<div class='prediction-letter' style='color: #D5DBDB;'>-</div><div class='status-box status-no-hand'>Hand Lost</div><div class='auto-status'>Finishing in {remaining}s...</div>",
            unsafe_allow_html=True,
        )
    else:
        prediction_placeholder.markdown(
            "<div class='prediction-letter' style='color: #D5DBDB;'>-</div><div class='status-box status-no-hand'>Waiting for Hand</div>",
            unsafe_allow_html=True,
        )

    current_word = st.session_state.word
    word_placeholder.markdown(
        f"<div class='word-display'><div class='word-text'>{current_word.upper()}</div></div>"
        if current_word
        else "<div class='word-display'><div class='word-text-empty'>Ready</div></div>",
        unsafe_allow_html=True,
    )
    if st.session_state.completed_words:
        sentence = " | ".join(st.session_state.completed_words)
        completed_placeholder.markdown(
            f"<div class='card' style='margin-top: 5px; border-color: #27AE60;'><div class='word-label' style='color: #27AE60;'>Completed Sentences</div><div style='font-size: 0.9rem; font-weight: 600;'>{sentence}</div></div>",
            unsafe_allow_html=True,
        )


render_live_status()

