import streamlit as st
import os


def process_video(input_path):
    import tempfile
    import cv2
    from ultralytics import YOLO

    model = YOLO("model.pt")

    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    h, w, _ = frame.shape

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (w, h))

    while ret:
        results = model(frame)[0]
        # Draw boxes (add your logic here)
        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    return output_path

st.title("Play a Local Video from Path")

# Input: Path to the video
video_path = process_video('test.mp4')
# Check and play the video
if video_path:
    if os.path.exists(video_path):
        st.video(video_path)
        st.success(f"Video loaded successfully: {video_path}")
    else:
        st.error("The video file was not found at the specified path.")

