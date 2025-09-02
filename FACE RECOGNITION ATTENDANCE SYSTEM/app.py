import os
import cv2
import av
import time
import numpy as np
import datetime
import streamlit as st
import face_recognition
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# -------------------------
# Page config & layout
# -------------------------
st.set_page_config(page_title="Face ID Access Control", layout="wide")
left, right = st.columns([2, 1])

with st.sidebar:
    st.header("Settings")
    train_dir = st.text_input("Training images folder", "./train")
    log_path = st.text_input("Log file", "log.txt")
    auto_refresh = st.checkbox("Auto-refresh log (1s)", value=True)
    top_n_log = st.slider("Show last N log rows", 10, 500, 100, 10)

# -------------------------
# Helpers
# -------------------------
@st.cache_resource(show_spinner=True)
def load_facebank(train_folder: str):
    folder = Path(train_folder)
    if not folder.exists():
        return [], []
    encodings, names = [], []
    for fname in sorted(os.listdir(folder)):
        fpath = folder / fname
        if not fpath.is_file():
            continue
        try:
            img = face_recognition.load_image_file(str(fpath))
            faces = face_recognition.face_encodings(img)
            if len(faces) > 0:
                encodings.append(faces[0])
                names.append(Path(fname).stem)  # filename (no extension) as name
        except Exception:
            # Ignore unreadable files
            pass
    return encodings, names

def append_log(name: str, path: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{name} {now}\n")

def read_log_tail(path: str, n: int = 100):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return [ln.rstrip("\n") for ln in lines[-n:]]

KNOWN_ENCODINGS, KNOWN_NAMES = load_facebank(train_dir)

# Session guard to avoid duplicate spam
if "last_logged_name" not in st.session_state:
    st.session_state.last_logged_name = None

# -------------------------
# Video processor
# -------------------------
class FaceAccessProcessor(VideoProcessorBase):
    def __init__(self):
        self.known_enc = KNOWN_ENCODINGS
        self.known_names = KNOWN_NAMES

    def _match_name(self, face_encoding):
        if not self.known_enc:
            return False, "Unknown", 1.0
        matches = face_recognition.compare_faces(self.known_enc, face_encoding)
        dists = face_recognition.face_distance(self.known_enc, face_encoding)
        i = int(np.argmin(dists))
        return (bool(matches[i]), self.known_names[i] if matches[i] else "Unknown", float(dists[i]))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        banner_text = "AUTHORIZATION DENIED"
        banner_color = (0, 0, 255)  # red
        granted_name = None
        granted_any = False

        for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
            matched, name, dist = self._match_name(enc)
            if matched:
                color = (0, 255, 0)  # green
                label = f"{name} | ENTRY GRANTED"
                granted_any = True
                granted_name = name
            else:
                color = (0, 0, 255)  # red
                label = "Unknown | AUTHORIZATION DENIED"

            cv2.rectangle(img, (left, top), (right, bottom), color, 3)
            cv2.putText(img, label, (left, max(20, top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        if granted_any:
            banner_text = "ENTRY GRANTED"
            banner_color = (0, 255, 0)
            # Log when name changes (prevents spamming while same face is in view)
            if granted_name and st.session_state.last_logged_name != granted_name:
                append_log(granted_name, log_path)
                st.session_state.last_logged_name = granted_name
        else:
            st.session_state.last_logged_name = None

        # Top banner
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, banner_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------
# Left: Camera
# -------------------------
with left:
    st.subheader("Camera")
    if len(KNOWN_ENCODINGS) == 0:
        st.warning(
            "No valid face encodings found in the training folder.\n"
            "Add images (one face per file) to the folder set in the sidebar."
        )

    webrtc_streamer(
        key="face-access",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceAccessProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("If prompted, click **Allow** to grant camera access in your browser.")

# -------------------------
# Right: Real-time log
# -------------------------
with right:
    st.subheader("Access Log")

    # Display latest log lines
    lines = read_log_tail(log_path, n=top_n_log)
    log_text = "\n".join(lines) if lines else "(No entries yet...)"
    st.text_area("Recent Entries", value=log_text, height=620, key="log_display")

    # Optional gentle auto-refresh (once per second)
    if auto_refresh:
        time.sleep(1)
        st.rerun()
