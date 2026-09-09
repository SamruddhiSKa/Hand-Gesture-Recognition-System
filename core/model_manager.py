import joblib
import streamlit as st
import logging
from pathlib import Path
from config.settings import MODEL_PATH

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="Loading gesture model...")
def load_model():
    """Load the trained model with Streamlit caching."""
    if not Path(MODEL_PATH).is_file():
        st.error("The gesture model is unavailable. Please contact the app owner.")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.exception("Model load error: %s", e)
        st.error("The gesture model could not be loaded. Please try again later.")
        return None
