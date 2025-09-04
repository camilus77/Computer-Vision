import streamlit as st
import os
import uuid
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import shutil
import subprocess
import imageio_ffmpeg


st.set_page_config(page_title="Dog Detection App", layout="wide")
st.title("🐶 Dog Detection App\nBy Ubong Camilus Ben")

MODEL_PATH = "../model/model.pt"
DEFAULT_CONF_THRESHOLD = 0.30

# -------------------------
# Model loading (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return YOLO(model_path)

# -------------------------
# Drawing utilities
# -------------------------
def draw_detections_bgr(bgr_image: np.ndarray, results, threshold: float = DEFAULT_CONF_THRESHOLD) -> np.ndarray:
    det = results[0]
    names = det.names
    if det.boxes is None or det.boxes.data is None:
        return bgr_image
    for x1, y1, x2, y2, score, class_id in det.boxes.data.tolist():
        if score >= threshold:
            cv2.rectangle(bgr_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            label = f"{names.get(int(class_id), 'obj').upper()} {score:.2f}"
            cv2.putText(bgr_image, label, (int(x1), max(0, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return bgr_image

# -------------------------
# Image processing
# -------------------------
def process_image(pil_image: Image.Image, model: YOLO, threshold: float = DEFAULT_CONF_THRESHOLD) -> Image.Image:
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = model(bgr)
    bgr_out = draw_detections_bgr(bgr, results, threshold=threshold)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)

# -------------------------
# Video processing
# -------------------------
def process_video(video_path: str, model: YOLO, threshold: float = DEFAULT_CONF_THRESHOLD) -> str:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError("Could not read the first frame from the uploaded video.")

    H, W = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25

    # unique temp file to avoid browser cache
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uuid.uuid4().hex}.mp4")
    tmp_out_path = tmp_out.name
    tmp_out.close()

    out = cv2.VideoWriter(tmp_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    while ret and frame is not None:
        results = model(frame)
        frame_out = draw_detections_bgr(frame, results, threshold=threshold)
        out.write(frame_out)
        ret, frame = cap.read()

    cap.release()
    out.release()
    return tmp_out_path

# -------------------------
# Make browser-friendly (H.264 + yuv420p + faststart)
# -------------------------
def make_browser_friendly(input_path: str) -> str:
    """
    Re-encode to a web-friendly MP4 using the FFmpeg binary bundled by imageio-ffmpeg.
    Returns the path to a playable MP4 (H.264 + yuv420p + faststart + AAC).
    Falls back to the original file if re-encode fails.
    """
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()  # <- bundled FFmpeg path

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "veryfast",
        "-crf", "23",
        # audio (make it browser-friendly too)
        "-c:a", "aac",
        "-b:a", "128k",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path
    except subprocess.CalledProcessError:
        return input_path

# -------------------------
# UI
# -------------------------
model = load_model(MODEL_PATH)
media_type = st.radio("What do you want to upload?", ("Image", "Video"), horizontal=True)

if media_type == "Image":
    uploaded_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_img is not None:
        try:
            image = Image.open(uploaded_img).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width =True)

            with st.spinner("Detecting dogs in image..."):
                processed_img = process_image(image, model=model, threshold=DEFAULT_CONF_THRESHOLD)

            with col2:
                st.image(processed_img, caption="Processed Image", use_container_width =True)

            # download
            buf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            processed_img.save(buf.name)
            with open(buf.name, "rb") as f:
                st.download_button("Download Processed Image", data=f.read(),
                                   file_name="processed_image.png", mime="image/png")
        except Exception as e:
            st.error(f"Image processing failed: {e}")

else:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1])
        tfile.write(uploaded_vid.read())
        tfile.flush()
        input_video_path = tfile.name

        col1, col2 = st.columns(2)

        with col1:
            st.video(input_video_path)
            st.markdown("**Original Video**")

        try:
            with st.spinner("Detecting dogs in video..."):
                raw_out_path = process_video(input_video_path, model=model, threshold=DEFAULT_CONF_THRESHOLD)
                playable_out_path = make_browser_friendly(raw_out_path)

            # Serve as bytes to ensure correct headers & avoid caching weirdness
            with open(playable_out_path, "rb") as f:
                video_bytes = f.read()

            with col2:
                st.video(video_bytes, format="video/mp4")
                st.markdown("**Processed Video**")

            # Download button for the browser-friendly file
            st.download_button("Download Processed Video",
                               data=video_bytes,
                               file_name="processed_video.mp4",
                               mime="video/mp4")

        except Exception as e:
            st.error(f"Video processing failed: {e}")
