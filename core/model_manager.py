import os
import joblib
import streamlit as st
import logging
from config.settings import MODEL_PATH

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="Loading gesture model...")
def load_model():
    """Load the trained model with Streamlit caching."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model not found at {MODEL_PATH}")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Model load error: {e}")
        st.error(f"⚠️ Model load error: {e}")
        return None
