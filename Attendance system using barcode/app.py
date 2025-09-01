# app.py
import time
import cv2
import numpy as np
import datetime as dt
from pathlib import Path
from pyzbar.pyzbar import decode
import streamlit as st

# -----------------------
# App config & constants
# -----------------------
st.set_page_config(page_title="Attendance Barcode System", layout="wide")
DATA_DIR = Path(".")
USERS_FILE = DATA_DIR / "users.txt"
LOG_FILE = DATA_DIR / "log.txt"

# -----------------------
# Helpers
# -----------------------
def load_users(path: Path) -> set:
    if not path.exists():
        path.write_text("")  # create empty file if missing
    with path.open("r", encoding="utf-8") as f:
        # strip newlines/whitespace and ignore short/empty lines
        return {line.strip() for line in f if len(line.strip()) > 0}

def append_log(person_id: str, path: Path):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{person_id} {ts}\n")

def read_log(path: Path, limit: int = 200) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    # Show last 'limit' lines (most recent at the bottom)
    return "".join(lines[-limit:])

def open_camera(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    # try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap

def close_camera():
    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.cap = None

# -----------------------
# Session state
# -----------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "logged_once" not in st.session_state:
    st.session_state.logged_once = set()  # avoid duplicate writes in one session

# -----------------------
# Sidebar (controls)
# -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    cam_index = st.number_input("Camera index", min_value=0, value=0, step=1)
    width = st.number_input("Width", min_value=320, value=640, step=80)
    height = st.number_input("Height", min_value=240, value=480, step=60)
    mirror = st.checkbox("Mirror (flip horizontally)", value=True)
    fps_limit = st.slider("Max FPS", 5, 60, 24)
    st.divider()
    if st.button("🧹 Clear attendance log"):
        LOG_FILE.write_text("")
        st.toast("Log cleared.")

# -----------------------
# Header (styled like your mockup)
# -----------------------
st.markdown(
    """
    <div style="border:2px solid #0a0a0a; padding:10px;">
      <div style="background:#e5e5e5; display:inline-block; padding:8px 16px; margin:8px auto;">
        <h1 style="margin:0; font-size:36px;">ATTENDANCE BARCODE SYSTEM</h1>
        <div style="font-weight:700; margin-top:2px;">By Ubong Camilus Ben</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Start / Stop controls
# -----------------------
col_btn1, col_btn2 = st.columns([1, 1])
start_clicked = col_btn1.button("▶️ Start")
stop_clicked = col_btn2.button("⏹ Stop")

if start_clicked and not st.session_state.running:
    # reset per-run state
    st.session_state.logged_once = set()
    close_camera()
    st.session_state.cap = open_camera(cam_index, width, height)
    if st.session_state.cap is None:
        st.error("Could not open the camera. Try a different index or close other apps using the webcam.")
    else:
        st.session_state.running = True
        st.toast("Streaming started.")

if stop_clicked and st.session_state.running:
    st.session_state.running = False
    close_camera()
    st.toast("Streaming stopped.")

# -----------------------
# Main two-column layout
# -----------------------
left, right = st.columns(2)

with left:
    st.markdown("<h2 style='text-align:center;'>SCAN HERE</h2>", unsafe_allow_html=True)
    scan_box = st.container(border=True)
    frame_area = scan_box.empty()
    status_area = st.empty()

with right:
    st.markdown("<h2 style='text-align:center;'>ATTENDANCE RECORD</h2>", unsafe_allow_html=True)
    log_box = st.container(border=True)
    log_area = log_box.empty()

# Preload users
users = load_users(USERS_FILE)

# -----------------------
# Streaming & decoding
# -----------------------
if st.session_state.running and st.session_state.cap is not None:
    cap = st.session_state.cap
    delay = 1.0 / max(fps_limit, 1)

    while st.session_state.running:
        ok, frame = cap.read()
        if not ok:
            status_area.error("Failed to read from camera. Stopping…")
            st.session_state.running = False
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        # Decode bar/QR codes
        codes = decode(frame)
        if len(codes):
            code = codes[0]  # handle first detected
            rect = code.rect
            polygon = code.polygon
            data = (code.data or b"").decode(errors="ignore").strip()

            # Draw shapes
            cv2.rectangle(frame, (rect.left, rect.top),
                          (rect.left + rect.width, rect.top + rect.height), (0, 0, 255), 3)
            if polygon:
                pts = np.array([(p.x, p.y) for p in polygon], dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

            if data in users:
                cv2.putText(frame, "ENTRY GRANTED", (rect.left, max(0, rect.top - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 3)
                # Log once per run for each ID
                if data not in st.session_state.logged_once:
                    append_log(data, LOG_FILE)
                    st.session_state.logged_once.add(data)
                    status_area.success(f"Logged: {data}")
            else:
                cv2.putText(frame, "AUTHORIZATION DENIED", (rect.left, max(0, rect.top - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # Convert BGR->RGB and show
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_area.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update log panel live
        log_text = read_log(LOG_FILE)
        log_area.text(log_text if log_text else "No records yet...")

        time.sleep(delay)

    # Cleanup when loop exits
    close_camera()
else:
    # Show the latest log even when not streaming
    log_text = read_log(LOG_FILE)
    log_area.text(log_text if log_text else "No records yet...")

