# app.py
import io
import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import easyocr as ocr

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="EasyOCR Streamlit App", layout="wide")
st.title("Interactive Image to Text Extractor\nBY UBONG CAMILUS BEN")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.01)
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False)
draw_mode = st.sidebar.radio("Selection mode", ["None (full image)", "Rectangle selection"], index=1)
canvas_display_width = st.sidebar.slider("Canvas display width (px)", 400, 1200, 800, 50)

# Cache the EasyOCR reader so it doesn't re-initialize on every rerun
@st.cache_resource(show_spinner=False)
def get_reader(lang=("en",), gpu=False):
    return ocr.Reader(list(lang), gpu=gpu)

reader = get_reader(gpu=use_gpu)

# ---------------------------
# Helpers
# ---------------------------
def load_image_to_cv2(file) -> np.ndarray:
    """Load uploaded file to a BGR cv2 image."""
    pil_img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_boxes_on_image(image_bgr: np.ndarray, ocr_results, threshold: float):
    img = image_bgr.copy()
    for bbox, text, conf in ocr_results:
        if conf >= threshold:
            # bbox is a list of four points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            p1 = tuple(map(int, bbox[0]))
            p3 = tuple(map(int, bbox[2]))
            cv2.rectangle(img, p1, p3, (0, 0, 255), 2)
            cv2.putText(img, text, (p1[0], max(0, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return img

def crop_with_rect(image_bgr: np.ndarray, rect, canvas_w: int, canvas_h: int):
    """
    rect: {"left": x, "top": y, "width": w, "height": h} in canvas coords.
    Need to map to original image size.
    """
    h, w = image_bgr.shape[:2]
    scale_x = w / canvas_w
    scale_y = h / canvas_h

    x = int(max(0, rect["left"] * scale_x))
    y = int(max(0, rect["top"] * scale_y))
    cw = int(rect["width"] * scale_x)
    ch = int(rect["height"] * scale_y)

    x2 = min(w, x + cw)
    y2 = min(h, y + ch)
    if x2 <= x or y2 <= y:
        return None
    return image_bgr[y:y2, x:x2]

def run_ocr(image_bgr: np.ndarray):
    # EasyOCR accepts RGB; but it also works with BGR numpy arrays.
    # Convert to RGB for consistency.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return reader.readtext(rgb)

# ---------------------------
# Layout
# ---------------------------
col_text, col_tools = st.columns([1, 2])

with col_tools:
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img_bgr = load_image_to_cv2(uploaded)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        # Drawable canvas (optional selection)
        st.markdown("**Optional:** draw a rectangle to OCR only that region.")
        canvas_result = None
        rect_shape = None

        if draw_mode == "Rectangle selection":
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",  # transparent fill
                stroke_width=3,
                stroke_color="#00FF00",
                background_image=Image.fromarray(img_rgb),
                update_streamlit=True,
                height=int(h * (canvas_display_width / w)),
                width=canvas_display_width,
                drawing_mode="rect",
                key="canvas",
            )
            # Extract first rectangle if any
            if canvas_result and canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                rects = [o for o in objects if o.get("type") == "rect"]
                if len(rects):
                    # Use the last drawn rectangle
                    rect_shape = rects[-1]
        else:
            # Just show the image (no canvas)
            st.image(img_rgb, caption="Uploaded image", use_container_width=True)

        # If a rectangle exists, crop; else run on full image
        region_for_ocr = img_bgr
        used_region = "Full image"
        if rect_shape is not None:
            rect = {
                "left": rect_shape.get("left", 0),
                "top": rect_shape.get("top", 0),
                "width": rect_shape.get("width", 0),
                "height": rect_shape.get("height", 0),
            }
            region = crop_with_rect(
                img_bgr, rect,
                canvas_w=canvas_result.image_data.shape[1] if canvas_result and canvas_result.image_data is not None else canvas_display_width,
                canvas_h=canvas_result.image_data.shape[0] if canvas_result and canvas_result.image_data is not None else int(h * (canvas_display_width / w)),
            )
            if region is not None and region.size > 0:
                region_for_ocr = region
                used_region = "Selected region"

        # Run OCR
        with st.spinner(f"Running OCR on: {used_region}…"):
            results = run_ocr(region_for_ocr)

        # Draw boxes on a preview image (only boxes >= threshold)
        boxed = draw_boxes_on_image(region_for_ocr, results, conf_threshold)
        st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB),
                 caption=f"OCR result preview ({used_region}) — boxes shown at ≥ {conf_threshold:.2f}",
                 use_container_width=True)

        # Collect recognised text above threshold
        extracted_lines = [text for (bbox, text, conf) in results if conf >= conf_threshold]
        joined_text = "\n".join(extracted_lines) if extracted_lines else "(No text found at the chosen threshold.)"

        # Download
        if extracted_lines:
            text_bytes = joined_text.encode("utf-8")
            st.download_button("Download extracted text", data=text_bytes,
                               file_name="extracted_text.txt", mime="text/plain")

    else:
        st.info("Upload an image to begin.")

with col_text:
    st.subheader("Extracted Text")
    st.caption("Text updates after you upload and/or draw a selection.")
    # Show a dynamic text box; if no upload yet, show placeholder.
    if 'joined_text' in locals():
        st.text_area("OCR Output", value=joined_text, height=500)
    else:
        st.text_area("OCR Output", value="", height=500, placeholder="Your extracted text will appear here…")

# ---------------------------
# Tips
# ---------------------------
with st.expander("Tips & Notes"):
    st.markdown(
        """
- Use the **confidence threshold** to filter out noisy results.
- Toggle **Rectangle selection** to limit OCR to a drawn region (handy for complex pages).
- If you have a capable GPU and proper drivers installed, tick **Use GPU** to speed things up.
- For best results, upload clear, high-contrast images.
        """
    )
