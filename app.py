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
from config.settings import WEBRTC_STUN_SERVERS, APP_TITLE, APP_DESCRIPTION, APP_ICON

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

# ─── Thread-safe Shared State ──────────────────────────────
class SharedState:
    """Thread-safe container to pass predictions from callback thread to main thread."""
    def __init__(self):
        self._result = {"prediction": "", "confidence": 0.0, "hand_detected": False}
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=5) # Reduced for faster response
        self._flip = True

    def update(self, raw_result):
        with self._lock:
            if raw_result["prediction"]:
                self._buffer.append(raw_result["prediction"])
            if self._buffer:
                most_common = Counter(self._buffer).most_common(1)[0][0]
                self._result = {
                    "prediction": most_common,
                    "confidence": raw_result["confidence"],
                    "hand_detected": raw_result["hand_detected"],
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
detector = None
try:
    detector = create_hands_detector()
except Exception as e:
    logging.exception("MediaPipe initialization error: %s", e)
    st.error("Hand tracking could not be initialized. Please try again later.")

# ─── Video Callback ─────────────────────────────────────────
def video_frame_callback(frame):
    try:
        img = frame.to_ndarray(format="bgr24")
        if detector is None or model is None:
            return frame
        # Get flip state from shared object
        should_flip = shared.get_flip()
        
        annotated_frame, result = process_frame(img, detector, model, flip=should_flip)
        shared.update(result)
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    except Exception as e:
        logging.error(f"Error in callback: {e}")
        return frame

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
        rtc_configuration=RTCConfiguration({"iceServers": WEBRTC_STUN_SERVERS}),
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
    )
    if not webrtc_ctx.state.playing:
        st.info("Camera is not active. Click Start and allow camera access in your browser. If access was denied, enable it in the browser site settings and reload the page.")

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
    
    clear_btn = st.button("✕ Clear All", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    completed_placeholder = st.empty()

# ─── Button Actions ─────────────────────────────────────────
if continue_btn and st.session_state.last_prediction:
    st.session_state.word += st.session_state.last_prediction
    st.rerun()

if end_btn and st.session_state.word:
    st.session_state.completed_words.append(st.session_state.word)
    st.session_state.word = ""
    st.rerun()

if clear_btn:
    st.session_state.word = ""
    st.session_state.completed_words = []
    st.rerun()

# ─── UI Updates ─────────────────────────────────────────────
if st.session_state.completed_words:
    sentence = " | ".join(st.session_state.completed_words)
    completed_placeholder.markdown(f"""
    <div class="card" style="margin-top: 5px; border-color: #27AE60;">
        <div class="word-label" style="color: #27AE60;">Sentences</div>
        <div style="font-size: 0.9rem; font-weight: 600;">{sentence}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Constants for Auto-Capture ────────────────────────────
HOLD_REQUIRED = 15.0 # Seconds to hold gesture
COOLDOWN_REQUIRED = 3.0 # Seconds before next capture
NO_HAND_TIMEOUT = 12.0 # Seconds of no hand to end word

is_active = webrtc_ctx.state.playing if webrtc_ctx else False

if is_active:
    while True:
        if not webrtc_ctx.state.playing: break
        
        now = time.time()
        result = shared.get()
        pred = result.get("prediction", "")
        hand_visible = result.get("hand_detected", False)

        # 1. Handle Cooldown
        if st.session_state.is_cooling_down:
            elapsed = now - st.session_state.cooldown_start_time
            if elapsed >= COOLDOWN_REQUIRED:
                st.session_state.is_cooling_down = False
            
        # 2. Handle Hand Visibility & Logic
        if hand_visible:
            st.session_state.no_hand_start_time = None # Reset auto-end timer
            
            if not st.session_state.is_cooling_down:
                if pred and pred == st.session_state.last_prediction:
                    if st.session_state.hold_start_time is None:
                        st.session_state.hold_start_time = now
                    
                    hold_elapsed = now - st.session_state.hold_start_time
                    progress = min(hold_elapsed / HOLD_REQUIRED, 1.0)
                    
                    if progress >= 1.0:
                        # TRIGGER CAPTURE (No st.rerun here)
                        st.session_state.word += pred
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
            
            absent_elapsed = now - st.session_state.no_hand_start_time
            if absent_elapsed >= NO_HAND_TIMEOUT and st.session_state.word:
                # TRIGGER AUTO-END (No st.rerun here)
                st.session_state.completed_words.append(st.session_state.word)
                st.session_state.word = ""
                st.session_state.no_hand_start_time = None

        # ─── UPDATE UI MANUALLY (Fluid Updates) ───────────────
        
        # Prediction Panel
        if hand_visible and pred:
            if st.session_state.is_cooling_down:
                prediction_placeholder.markdown(f"""
                <div class="prediction-letter" style="color: #2E86C1;">{pred}</div>
                <div class="cooldown-status" style="color: #27AE60;">ADDED! {pred}</div>
                """, unsafe_allow_html=True)
            else:
                hold_elapsed = (now - st.session_state.hold_start_time) if st.session_state.hold_start_time else 0
                pct = min(int((hold_elapsed / HOLD_REQUIRED) * 100), 100)
                remaining = max(0, int(HOLD_REQUIRED - hold_elapsed))
                prediction_placeholder.markdown(f"""
                <div class="prediction-letter">{pred}</div>
                <div class="auto-status">Hold {remaining}s more...</div>
                <div class="progress-container"><div class="progress-fill" style="width: {pct}%;"></div></div>
                """, unsafe_allow_html=True)
        elif hand_visible:
            prediction_placeholder.markdown("""
            <div class="prediction-letter" style="color: #D5DBDB;">?</div>
            <div class="status-box status-no-hand">Position Hand...</div>
            """, unsafe_allow_html=True)
        else:
            if st.session_state.word:
                absent_e = now - st.session_state.no_hand_start_time
                rem = max(0, int(NO_HAND_TIMEOUT - absent_e))
                prediction_placeholder.markdown(f"""
                <div class="prediction-letter" style="color: #D5DBDB;">—</div>
                <div class="status-box status-no-hand">Hand Lost</div>
                <div class="auto-status">Finishing in {rem}s...</div>
                """, unsafe_allow_html=True)
            else:
                prediction_placeholder.markdown("""
                <div class="prediction-letter" style="color: #D5DBDB;">—</div>
                <div class="status-box status-no-hand">Waiting for Hand</div>
                """, unsafe_allow_html=True)

        # Word Builder (Update instantly in loop)
        curr = st.session_state.word
        if curr:
            word_placeholder.markdown(f"<div class='word-display'><div class='word-text'>{curr.upper()}</div></div>", unsafe_allow_html=True)
        else:
            word_placeholder.markdown("<div class='word-display'><div class='word-text-empty'>Ready</div></div>", unsafe_allow_html=True)

        # History / Sentences (Update instantly in loop)
        if st.session_state.completed_words:
            sentence = " | ".join(st.session_state.completed_words)
            completed_placeholder.markdown(f"""
            <div class="card" style="margin-top: 5px; border-color: #27AE60;">
                <div class="word-label" style="color: #27AE60;">Completed Sentences</div>
                <div style="font-size: 0.9rem; font-weight: 600;">{sentence}</div>
            </div>
            """, unsafe_allow_html=True)
        
        time.sleep(0.1)
else:
    prediction_placeholder.markdown("<div class='status-box status-inactive'>Waiting for Camera</div>", unsafe_allow_html=True)
    word_placeholder.markdown("<div class='word-display'><div class='word-text-empty'>—</div></div>", unsafe_allow_html=True)

