# app.py
import streamlit as st
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import supervision as sv
from Speedometer_helper import *

st.set_page_config(page_title="🚦 Traffic Monitor", layout="wide")
st.title("🚦 Advanced Traffic Monitoring and Speed Estimation")
st.write("Upload a traffic video to detect vehicles, track them, and estimate speed.")

# File uploader
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_video_path = "input_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Video info
    video_info = sv.VideoInfo.from_video_path(temp_video_path)
    FPS = video_info.fps

    # Custom colour palette
    colors = ("#007fff", "#0072e6", "#0066cc", "#0059b3", "#004c99", "#004080", "#003366", "#00264d")
    color_palette = sv.ColorPalette(list(map(sv.Color.from_hex, colors)))

    # Polygon zone
    poly = np.array([(0, 410), (1920, 410), (1920, 900), (0, 900)])
    zone = sv.PolygonZone(poly, (sv.Position.TOP_CENTER, sv.Position.BOTTOM_CENTER))

    # Annotators
    bbox_annotator = sv.BoxAnnotator(
        color=color_palette, thickness=2, color_lookup=sv.ColorLookup.TRACK
    )
    trace_annotator = sv.TraceAnnotator(
        color=color_palette, position=sv.Position.CENTER,
        thickness=2, trace_length=FPS, color_lookup=sv.ColorLookup.TRACK
    )
    label_annotator = sv.RichLabelAnnotator(
        color=color_palette, border_radius=2, font_size=16,
        color_lookup=sv.ColorLookup.TRACK, text_padding=6
    )

    # Perspective mapping
    image_pts = [(800, 410), (1125, 410), (1920, 850), (0, 850)]
    world_pts = [(0, 0), (32, 0), (32, 140), (0, 140)] 
    mapper = Cam2WorldMapper()
    mapper.find_perspective_transform(image_pts, world_pts)

    # YOLO + Speedometer
    yolo = YOLO("yolo11m.pt", task="detect")
    speedometer = Speedometer(mapper, FPS)

    output_video = "processed_video.mp4"
    width, height = video_info.resolution_wh
    width, height = round(width / 32) * 32, round(height / 32) * 32
    classes = [2, 5, 7]  # Car, Bus, Truck
    conf = 0.4

    st.info("⚡ Processing video, please wait...")
    with sv.VideoSink(output_video, video_info) as sink:
        for frame in sv.get_video_frames_generator(temp_video_path):
            result = yolo.track(
                frame,
                classes=classes,
                conf=conf,
                imgsz=(height, width),
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )
            detection = sv.Detections.from_ultralytics(result[0])
            detection = detection[zone.trigger(detections=detection)]

            trace_ids = detection.tracker_id
            speeds, labels = [], []

            for trace_id in trace_ids:
                image_trace = trace_annotator.trace.get(trace_id)
                speedometer.update_with_trace(int(trace_id), image_trace)
                current_speed = speedometer.get_current_speed(int(trace_id))
                speeds.append(current_speed)
                labels.append(f"#{trace_id} {current_speed} km/h")

            frame = cv.cvtColor(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
            frame = bbox_annotator.annotate(frame, detection)  
            frame = trace_annotator.annotate(frame, detection)  
            frame = label_annotator.annotate(frame, detection, labels=labels)  
            sink.write_frame(frame)

    st.success("✅ Processing completed!")
    st.video(output_video)
    with open(output_video, "rb") as f:
        st.download_button("⬇️ Download Processed Video", f, file_name="processed_video.mp4")
