# app.py
import streamlit as st
import cv2
import pandas as pd
import numpy as np
import math
import time
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Traffic App", page_icon="🚦", layout="wide")
st.title("🚦 Traffic Speed Detection App")

# ------------------- Upload video -------------------
uploaded_file = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

# ------------------- Load model -------------------
model = YOLO("yolo11n.pt")
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane','bus', 'train','truck']

# ------------------- Tracker (UNCHANGED) -------------------
def tracker(objects_rect, center_points={}, id_count=0):
    # Objects boxes and ids
    objects_bbs_ids = []

    # Get center point of new object
    for rect in objects_rect:
        x, y, w, h = rect
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2

        # Find out if that object was detected already
        same_object_detected = False
        for id, pt in center_points.items():
            dist = math.hypot(cx - pt[0], cy - pt[1])

            if dist < 35:
                center_points[id] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, id])
                same_object_detected = True
                break

        # New object is detected we assign the ID to that object
        if same_object_detected is False:
            center_points[id_count] = (cx, cy)
            objects_bbs_ids.append([x, y, w, h, id_count])
            id_count += 1

    # Clean the dictionary by center points to remove IDS not used anymore
    new_center_points = {}
    for obj_bb_id in objects_bbs_ids:
        _, _, _, _, object_id = obj_bb_id
        center = center_points[object_id]
        new_center_points[object_id] = center

    # Update dictionary with IDs not used removed
    center_points = new_center_points.copy()
    return objects_bbs_ids

# ------------------- Run button -------------------
if uploaded_file is not None:
    if st.button("▶️ Process Video"):
        # Save uploaded to a temp file for OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
            st.stop()

        # Prepare output writer (same size as display: 1020x500)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (1020, 500))

        count = 0
        down = {}
        up = {}
        detected = set()

        red_line_y = 200
        blue_line_y = 250
        offset = 4

        # Colors (BGR)
        red_color = (0, 0, 255)
        blue_color = (255, 0, 0)
        yellow_color = (0, 255, 255)
        green_color = (0, 255, 0)
        white_color = (255, 255, 255)

        frame_placeholder = st.empty()
        progress = st.progress(0, text="Processing…")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            frame = cv2.resize(frame, (1020, 500))

            # get prediction for each frame
            results = model.predict(frame)
            a = results[0].boxes.data
            a = a.detach().cpu().numpy()
            px = pd.DataFrame(a).astype("float")

            list_boxes = []
            for row in range(len(px)):
                x1, y1, x2, y2, _, cl_id = px.iloc[row,:].values.flatten().tolist()
                c = class_list[int(cl_id)]
                if c == 'car':
                    # keep exactly as you wrote: pass x1,y1,x2,y2 into tracker
                    list_boxes.append([x1, y1, x2, y2])

            bbox_id = tracker(list_boxes)
            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2

                # ------------------- HIGHLIGHT WHEN SPEED IS ABOUT TO POP UP -------------------
                if id in down or id in up:
                    cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), green_color, 2)
                # -------------------------------------------------------------------------------

                # Red line crossing then Blue
                if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                    down[id] = time.time()
                if id in down:
                    if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                        time1 = down[id]
                        time2 = time.time()
                        if time1 == time2:
                            speed = 0
                        else:
                            speed = 50 / (time2 - time1)
                        detected.add(id)

                        cv2.circle(frame, (cx, cy), 4, red_color, -1)
                        cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), red_color, 2)
                        cv2.putText(frame, ('Speed:' + str(round(speed*3.6, 2)) + 'km/hr'),
                                    (int(x4), int(y3)), cv2.FONT_HERSHEY_COMPLEX, 0.5, yellow_color, 1)
                        cv2.putText(frame, ('No of vehicles with speed detected - ') + str(len(detected)),
                                    (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1, cv2.LINE_AA)

                # Blue line crossing then Red
                if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                    up[id] = time.time()
                if id in up:
                    if red_line_y > (cy - offset):
                        time1 = up[id]
                        time2 = time.time()
                        if time1 == time2:
                            speed = 0
                        else:
                            speed = 50 / (time2 - time1)

                        detected.add(id)
                        cv2.circle(frame, (cx, cy), 4, red_color, -1)
                        cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), red_color, 2)
                        cv2.putText(frame, ('Speed:' + str(round(speed*3.6, 2)) + 'km/hr'),
                                    (int(x4), int(y3)), cv2.FONT_HERSHEY_COMPLEX, 0.5, yellow_color, 1)


            # write frame to output video
            out_writer.write(frame)

            # show current frame in the app 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if total_frames > 0:
                progress.progress(min(1.0, count / total_frames),
                                  text=f"Processing… {count}/{total_frames} frames")
            else:
                progress.progress((count % 100) / 100, text=f"Processing… frame {count}")

        cap.release()
        out_writer.release()

        st.success("Processing finished ✅")

        # Provide download button for processed video
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download Processed Video",
                data=f.read(),
                file_name="processed_traffic.mp4",
                mime="video/mp4",
                use_container_width=True
            )
