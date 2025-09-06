# app.py
import streamlit as st
import cv2
import math
import pandas as pd
import numpy as np
import tempfile
import time
from ultralytics import YOLO

st.set_page_config(page_title="Car Counter (YOLO + Tracker)", layout="wide")

# ---------- Sidebar controls ----------
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], index=0)
conf_thres = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
iou_thres = st.sidebar.slider("IOU threshold", 0.1, 0.9, 0.45, 0.05)

red_line_y = st.sidebar.slider("Red line Y", 50, 450, 300, 2)
blue_line_y = st.sidebar.slider("Blue line Y", 50, 450, 400, 2)
line_offset = st.sidebar.slider("Line hit offset (px)", 1, 15, 4, 1)

target_class = st.sidebar.selectbox("Detect class", ["car", "bus", "truck"], index=0)

uploaded_video = st.sidebar.file_uploader("Upload a video (MP4/MOV/AVI)", type=["mp4", "mov", "avi"])

# ---------- Session state ----------
def _init_state():
    if "center_points" not in st.session_state:
        st.session_state.center_points = {}  # id -> (cx, cy)
    if "id_count" not in st.session_state:
        st.session_state.id_count = 0
    if "down_map" not in st.session_state:   # ids that touched red first
        st.session_state.down_map = {}
    if "up_map" not in st.session_state:     # ids that touched blue first
        st.session_state.up_map = {}
    if "counter_down" not in st.session_state:
        st.session_state.counter_down = set()
    if "counter_up" not in st.session_state:
        st.session_state.counter_up = set()
    if "running" not in st.session_state:
        st.session_state.running = False

_init_state()

# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return YOLO(name)

model = load_model(model_name)

CLASS_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane','bus', 'train','truck']

# ---------- Tracker  ----------
def tracker(objects_rect):
    """
    objects_rect: list of [x1, y1, x2, y2] (float)
    Returns: list of [x1, y1, x2, y2, id]
    """
    center_points = st.session_state.center_points
    id_count = st.session_state.id_count
    objects_bbs_ids = []

    for rect in objects_rect:
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)

        same_object_detected = False
        for obj_id, pt in center_points.items():
            dist = math.hypot(cx - pt[0], cy - pt[1])
            if dist < 35:
                center_points[obj_id] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, obj_id])
                same_object_detected = True
                break

        if not same_object_detected:
            center_points[id_count] = (cx, cy)
            objects_bbs_ids.append([x1, y1, x2, y2, id_count])
            id_count += 1

    # Clean out stale IDs
    new_center_points = {}
    for _, _, _, _, object_id in objects_bbs_ids:
        new_center_points[object_id] = center_points[object_id]

    st.session_state.center_points = new_center_points
    st.session_state.id_count = id_count
    return objects_bbs_ids

# ---------- Video helpers ----------
def write_temp_file(uploaded):
    """Save uploaded video to a temp file and return path."""
    if uploaded is None:
        return None
    suffix = "." + uploaded.name.split(".")[-1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded.read())
    tfile.flush()
    return tfile.name

def draw_overlay(frame, down_ct, up_ct):
    text_color = (255,255,255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    green_color = (0, 255, 0)

    h, w = frame.shape[:2]

    cv2.putText(frame, f'going down - {down_ct}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f'going up   - {up_ct}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 2, cv2.LINE_AA)
    return frame

def process_and_stream(video_path: str, conf=0.35, iou=0.45):
    # Reset counts per run
    st.session_state.center_points = {}
    st.session_state.id_count = 0
    st.session_state.down_map = {}
    st.session_state.up_map = {}
    st.session_state.counter_down = set()
    st.session_state.counter_up = set()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        return

    placeholder = st.empty()
    metrics1, metrics2 = st.columns(2)
    fps_placeholder = st.empty()

    prev_time = time.time()
    frame_cnt = 0

    while st.session_state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1

        # Resize for consistent geometry 
        frame = cv2.resize(frame, (1020, 500))

        # YOLO inference
        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        boxes = results[0].boxes

        # Convert detections to DataFrame-like ndarray: [x1,y1,x2,y2,conf,cls]
        if boxes is None or boxes.data is None or len(boxes.data) == 0:
            det_arr = np.zeros((0, 6), dtype=float)
        else:
            det_arr = boxes.data.detach().cpu().numpy()

        df = pd.DataFrame(det_arr, columns=["x1","y1","x2","y2","conf","cls"])

        # Filter by target class (car/bus/truck)
        wanted = []
        for _, r in df.iterrows():
            cls_idx = int(r["cls"])
            if 0 <= cls_idx < len(CLASS_LIST) and CLASS_LIST[cls_idx] == target_class:
                wanted.append([float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])])

        # Track across frames
        bbox_id = tracker(wanted)
        for x1, y1, x2, y2, oid in bbox_id:
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)

            # draw bbox and id
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{target_class} #{oid}", (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0,255,255), -1)

            # counting logic (down: red -> blue)
            if red_line_y < (cy + line_offset) and red_line_y > (cy - line_offset):
                st.session_state.down_map[oid] = cy

            if oid in st.session_state.down_map:
                if blue_line_y < (cy + line_offset) and blue_line_y > (cy - line_offset):
                    st.session_state.counter_down.add(oid)

            # counting logic (up: blue -> red)
            if blue_line_y < (cy + line_offset) and blue_line_y > (cy - line_offset):
                st.session_state.up_map[oid] = cy

            if oid in st.session_state.up_map:
                if red_line_y > (cy - line_offset) and red_line_y < (cy + line_offset):
                    st.session_state.counter_up.add(oid)

        down_ct = len(st.session_state.counter_down)
        up_ct = len(st.session_state.counter_up)

        frame = draw_overlay(frame, down_ct, up_ct)

        # Update UI
        bgr = frame
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        placeholder.image(rgb, use_container_width=True)


        # FPS
        now = time.time()
        if now - prev_time >= 0.5:
            fps = frame_cnt / (now - prev_time)
            fps_placeholder.caption(f"FPS: {fps:.1f}")
            prev_time = now
            frame_cnt = 0

    cap.release()

# ---------- Layout ----------
st.title("🚗 Car/Bus/Truck Counter (YOLO)\n By **Ubong Ben**")
st.write(
    "Upload a traffic video or use a sample, then click **Start**. "
    "The app tracks vehicles - notes and counts ghem based on directionn."
)

left, right = st.columns([2, 1])

with right:
    use_sample = st.toggle("Use sample video", value=True, help="Uses a small built-in demo if you don't upload.")
    start = st.button("▶️ Start")
    stop = st.button("⏹️ Stop")

with left:
    frame_area = st.empty()

# ---------- Run control ----------
if start:
    st.session_state.running = True
    video_path = None

    if uploaded_video is not None:
        video_path = write_temp_file(uploaded_video)
    elif use_sample:
        # Provide your own local file name here (ensure it's available)
        video_path = "traffic2.mp4"
    else:
        st.warning("Please upload a video or enable the sample.")
    
    if video_path:
        process_and_stream(video_path, conf=conf_thres, iou=iou_thres)

if stop:
    st.session_state.running = False
