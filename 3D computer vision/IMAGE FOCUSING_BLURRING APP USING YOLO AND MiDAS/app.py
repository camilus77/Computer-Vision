import io
import cv2
import gc
import torch
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Depth Focus – Two Frames", layout="wide")
st.title("IMAGE BLURRING/FOCUSING APP \nUbong Camilus Ben")
st.caption("Image Blurring/Focusing app")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_depth_model(depth_model_type: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("intel-isl/MiDaS", depth_model_type).to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if depth_model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return model, transform, device

@st.cache_resource(show_spinner=True)
def load_seg_model(weights: str):
    return YOLO(weights)

def to_uint8_mask(mask_bool_or_float: np.ndarray) -> np.ndarray:
    m = mask_bool_or_float.astype(np.uint8)
    if m.max() == 1:
        m = m * 255
    return m

def yolo_masks_to_image_size(masks: np.ndarray, H: int, W: int) -> np.ndarray:
    if masks.shape[1] != H or masks.shape[2] != W:
        out = []
        for m in masks:
            m_u8 = (m * 255).astype(np.uint8)
            m_rs = cv2.resize(m_u8, (W, H), interpolation=cv2.INTER_NEAREST)
            out.append((m_rs > 127).astype(np.uint8))
        return np.stack(out, axis=0)
    return (masks > 0.5).astype(np.uint8)

def compute_depth_map(midas, transform, device, rgb: np.ndarray) -> np.ndarray:
    """Return a colourised depth map (uint8 RGB) and a plain grayscale depth array (uint8)."""
    H, W = rgb.shape[:2]
    inp = transform(rgb).to(device)      # ✅ Pass NumPy array directly
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth = pred.detach().cpu().numpy()
    del pred, inp; gc.collect()

    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_u8 = depth_norm.astype(np.uint8)

    # Colourise using matplotlib's 'inferno' colormap for a vivid heatmap
    cmap = plt.get_cmap("inferno")
    depth_rgb = (cmap(depth_u8/255.0)[:, :, :3] * 255).astype(np.uint8)
    return depth_u8, depth_rgb

def apply_portrait_effect(rgb: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(rgb, (21, 21), 0)
    mask_3 = np.dstack([final_mask]*3)
    inv_3  = cv2.bitwise_not(mask_3)
    fg = cv2.bitwise_and(rgb, mask_3)
    bg = cv2.bitwise_and(blurred, inv_3)
    return cv2.add(fg, bg)

def resize_max_side(img_rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Resize image so the longest side <= max_side to save memory."""
    h, w = img_rgb.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img_rgb
    scale = max_side / side
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    depth_model_type = st.selectbox(
        "Depth model",
        options=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
        index=0,
        help="Smaller model = less memory."
    )
    seg_weights = st.selectbox(
        "YOLOv8-Seg weights",
        options=["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"],
        index=0
    )
    conf_thres = st.slider("YOLO confidence", 0.05, 0.75, 0.25, 0.05)
    pad_depth = st.slider("Depth band padding (±)", 0, 40, 10, 1)
    max_side = st.select_slider(
        "Max processing side (px)",
        options=[512, 640, 768, 960, 1280],
        value=960
    )
    imgsz = st.select_slider(
        "YOLO inference size (px)",
        options=[512, 640, 768, 960],
        value=640
    )

# ---------------------------
# Layout: two frames (columns)
# ---------------------------
left_frame, right_frame = st.columns(2, gap="large")

# ---------------------------
# Input (left column)
# ---------------------------
with left_frame:
    st.subheader("📷 Original Image")
    uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])
    url = st.text_input("...or paste an image URL")

    if uploaded:
        image_bytes = uploaded.read()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil)
    elif url:
        try:
            import requests
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
            rgb = np.array(pil)
        except Exception as e:
            st.error(f"Failed to fetch image: {e}")
            st.stop()
    else:
        st.info("Upload an image or paste a URL to proceed.")
        st.stop()

    rgb = resize_max_side(rgb, max_side)
    H, W = rgb.shape[:2]
    st.image(rgb, caption="Original (resized for processing)", use_container_width=True)

# ---------------------------
# Load models
# ---------------------------
with st.spinner("Loading models..."):
    midas, midas_transform, device = load_depth_model(depth_model_type)
    yolo = load_seg_model(seg_weights)

# ---------------------------
# Depth estimation
# ---------------------------
with st.spinner("Estimating depth..."):
    depth_u8, depth_rgb = compute_depth_map(midas, midas_transform, device, rgb)

# ---------------------------
# Instance segmentation
# ---------------------------
with st.spinner("Running instance segmentation..."):
    res_list = yolo.predict(source=rgb, imgsz=imgsz, conf=conf_thres, verbose=False)
    if not res_list:
        st.error("No YOLO results.")
        st.stop()
    res = res_list[0]
    if res.masks is None or res.masks.data is None or len(res.masks.data) == 0:
        st.error("No instance masks detected.")
        st.stop()
    masks = res.masks.data.cpu().numpy()
    masks = yolo_masks_to_image_size(masks, H, W)

# ---------------------------
# Closest object by depth
# ---------------------------
closest_idx, max_avg = None, -1.0
for i in range(masks.shape[0]):
    m = masks[i]
    vals = depth_u8[m > 0]
    if vals.size == 0:
        continue
    avg = float(np.mean(vals))
    if avg > max_avg:
        max_avg = avg
        closest_idx = i

if closest_idx is None:
    st.error("Could not determine the closest instance.")
    st.stop()

closest_mask = masks[closest_idx].astype(np.uint8)
closest_mask_u8 = to_uint8_mask(closest_mask)

obj_vals = depth_u8[closest_mask > 0]
mn, mx = int(obj_vals.min()), int(obj_vals.max())
lower, upper = max(mn - pad_depth, 0), min(mx + pad_depth, 255)
depth_band = cv2.inRange(depth_u8, lower, upper)
final_mask = cv2.bitwise_and(closest_mask_u8, depth_band)
final_mask = to_uint8_mask(final_mask)

# ---------------------------
# Compose final blurred result
# ---------------------------
final = apply_portrait_effect(rgb, final_mask)

# ---------------------------
# Right column: Depth + Blurred
# ---------------------------
with right_frame:
    st.subheader("🖼️ Depth / Blurred")
    tab_depth, tab_blurred = st.tabs(["Colourised Depth Map", "Blurred Result"])
    with tab_depth:
        st.image(depth_rgb, caption="Depth Map (colourised)", use_container_width=True)
    with tab_blurred:
        st.image(final, caption="Blurred Result (Nearest object in focus)", use_container_width=True)

    out_bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", out_bgr)
    if ok:
        st.download_button(
            "⬇️ Download Blurred Result (PNG)",
            data=buf.tobytes(),
            file_name="depth_focus_output.png",
            mime="image/png"
        )
