import joblib
import streamlit as st
import logging
from pathlib import Path
from config.settings import MODEL_PATH, NUM_FEATURES

logger = logging.getLogger(__name__)
EXPECTED_CLASSES = tuple("0123456789abcdefghijklmnopqrstuvwxyz")


def model_contract_error(model):
    """Return a diagnostic when a serialized model violates the inference contract."""
    if getattr(model, "n_features_in_", None) != NUM_FEATURES:
        return f"Expected {NUM_FEATURES} input features"
    model_classes = tuple(str(label) for label in getattr(model, "classes_", ()))
    if model_classes != EXPECTED_CLASSES:
        return "Expected the ordered gesture labels 0-9 and a-z"
    if not callable(getattr(model, "predict_proba", None)):
        return "Expected a classifier that supports predict_proba"
    return None

@st.cache_resource(show_spinner="Loading gesture model...")
def load_model():
    """Load the trained model with Streamlit caching."""
    if not Path(MODEL_PATH).is_file():
        st.error("The gesture model is unavailable. Please contact the app owner.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        contract_error = model_contract_error(model)
        if contract_error:
            logger.error("Incompatible gesture model: %s", contract_error)
            st.error("The gesture model is incompatible with this application.")
            return None
        return model
    except Exception as e:
        logger.exception("Model load error: %s", e)
        st.error("The gesture model could not be loaded. Please try again later.")
        return None
