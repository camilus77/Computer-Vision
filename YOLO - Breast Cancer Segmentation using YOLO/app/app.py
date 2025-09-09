import os
import tempfile
import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Mask Segmentation", page_icon="🩺", layout="centered")
st.title("BREAST CANCER SEGMENTATION USING YOLO (Mask Segmentation)\nBY UBONG CAMILUS BEN")

with st.sidebar:
    st.header("⚙️ Settings")
    model_path = "..\\models\\last.pt"
    st.markdown("---")
    uploaded = st.file_uploader("Pick a local image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    run_btn = st.button("▶️ Run Segmentation", type="primary")

# Will display results here
img_col, out_col = st.columns(2)

if run_btn:
    if not model_path or not os.path.exists(model_path):
        st.error("Model file not found. Check the path in the sidebar.")
        st.stop()

    if uploaded is None:
        st.warning("Please upload an image.")
        st.stop()

    # Persist the uploaded file to a real local path so your code can use it
    suffix = os.path.splitext(uploaded.name)[-1].lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp.flush()
        image_path = tmp.name  # <-- feed this to your code exactly

    # --- YOUR ORIGINAL LOGIC (unchanged) ---
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    H, W, _ = img.shape
    results = model(img)
    for result in results:
        for j, mask in enumerate(result.masks.data):
            mask = mask.numpy() * 255
            mask = cv2.resize(mask, (W, H))
            cv2.imwrite('./output.png', mask)
    # ---------------------------------------

    # Show input image
    with img_col:
        st.subheader("Input")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=os.path.basename(image_path), use_container_width=True)

    # Show saved mask (your script always writes ./output.png)
    out_path = "./output.png"
    with out_col:
        st.subheader("Output Mask [white parts show cancer]")
        if os.path.exists(out_path):
            st.image(out_path, caption="output.png", use_container_width=True)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download output.png", f, file_name="output.png")
        else:
            st.info("No mask file found. Ensure the model produced masks and the script wrote ./output.png")

st.markdown("---")
