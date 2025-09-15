# app.py
import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------
# Page setup & styling
# -------------------------
st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="wide")
st.markdown("""
<style>
/* Clean, card-like containers */
.reportview-container .main .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1250px;}
.card {background: #111827; border-radius: 18px; padding: 18px 20px; box-shadow: 0 10px 30px rgba(0,0,0,.25); border: 1px solid #1f2937;}
h1, h2, h3, h4 {letter-spacing: .2px;}
.kpi {font-size: 44px; font-weight: 800; margin: 0;}
.badge {display: inline-block; padding: 6px 10px; border-radius: 999px; background: #111827; border: 1px solid #374151; color: #e5e7eb; font-size: 12px;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Skin Lesion Classification — Demo App")
st.caption("This is a transparent prediction of your skin disease. **Not medical advice.**")

# -------------------------
# Your original mapping (unchanged)
# -------------------------
label_dictionary = {
    0: 'actinic keratosis',
    1: 'basal cell carcinoma',
    2: 'dermatofibroma',
    3: 'melanoma',
    4: 'nevus',
    5: 'pigmented benign keratosis',
    6: 'seborrheic keratosis',
    7: 'squamous cell carcinoma',
    8: 'vascular lesion'
}
labels_ordered = [label_dictionary[i] for i in range(len(label_dictionary))]

# -------------------------
# Load model once (no logic change, just caching)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str):
    return load_model(model_path)

with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model file", value="skin_disease_model.h5")
    show_probs = st.toggle("Show probability chart", value=True)
    st.markdown("---")
    st.markdown("**Input options**")
    uploaded = st.file_uploader("Upload lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    sample_path = st.text_input("…or use a local path", value="")

# Try load model
model = None
model_ok = False
err_msg = None
if model_path and os.path.exists(model_path):
    try:
        model = load_trained_model(model_path)
        model_ok = True
    except Exception as e:
        err_msg = f"Failed to load model: {e}"
else:
    err_msg = "Model file not found. Ensure 'skin_disease_model.h5' is in this folder."

# -------------------------
# Helper: your exact preprocessing logic
# (target_size=(75, 100), per-image mean/std normalisation)
# -------------------------
def preprocess_pil(pil_img):
    img = pil_img.resize((100, 75))  # (width, height) -> matches target_size=(75, 100)
    # BUT keras load_img uses (height, width); your code used target_size=(75, 100)
    # We keep the same end shape (75, 100, 3)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = (arr - np.mean(arr)) / np.std(arr)
    return arr

# Select source image
input_image = None
src_note = None

if uploaded is not None:
    try:
        input_image = Image.open(uploaded).convert("RGB")
        src_note = f"Uploaded file: **{uploaded.name}**"
    except Exception as e:
        st.error(f"Could not open uploaded image: {e}")

elif sample_path.strip():
    if os.path.exists(sample_path):
        try:
            input_image = Image.open(sample_path).convert("RGB")
            src_note = f"Local path: **{sample_path}**"
        except Exception as e:
            st.error(f"Could not open local image: {e}")
    else:
        st.warning("Local path not found.")

# -------------------------
# Main UI
# -------------------------
c1, c2 = st.columns([1, 1])

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Input")
    if input_image is None:
        st.info("Upload an image or provide a local path from the sidebar to run a prediction.")
        st.image("https://images.unsplash.com/photo-1516637090014-cb1ab0d08fc7?q=80&w=800&auto=format&fit=crop", use_container_width=True)
    else:
        st.caption(src_note or "")
        st.image(input_image, caption="Input image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Prediction")
    if not model_ok:
        st.error(err_msg or "Model not loaded.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if input_image is None:
            st.warning("Awaiting image…")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Running inference…"):
                arr = preprocess_pil(input_image)  # same mean/std normalisation + shape
                predicted_vector = model.predict(arr)
                predicted_index = int(np.argmax(predicted_vector))
                predicted_class = label_dictionary[predicted_index]
                confidence = float(np.max(tf.nn.softmax(predicted_vector, axis=-1)))

            # KPI-style result
            st.markdown(f"Predicted class:")
            st.markdown(f"<div class='kpi'>{predicted_class.title()}</div>", unsafe_allow_html=True)
            st.caption(f"Confidence: **{confidence:.2%}**")

            # Probabilities chart (optional)
            if show_probs:
                st.markdown("---")
                st.markdown("**Class probabilities**")
                fig, ax = plt.subplots(figsize=(8, 4))
                probs = (predicted_vector.flatten()).astype(float)
                ax.bar(labels_ordered, probs)
                ax.set_xticklabels(labels_ordered, rotation=45, ha="right")
                ax.set_ylabel("Probability")
                ax.set_ylim(0, 1)
                st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Disclosure
# -------------------------
st.markdown("---")
st.warning(
    "This is a demo. "
    "**It is not a medical device and must not be used for diagnosis or treatment.**"
)

st.caption("Dependencies: `streamlit`, `tensorflow`, `Pillow`, `matplotlib`, `numpy`")
