import os
import cv2
import time
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# ============================================================
# ------------------------- CONFIG ---------------------------
# ============================================================

CUSTOM_MODEL_PATH = None           # Path to a custom YOLOv5 model (.pt).  None -> use pretrained 'yolov5s'
CONF_THRESHOLD   = 0.5             # Minimum detection confidence to keep a box
TARGET_CLASSES   = None            # None -> track ALL classes; or e.g., ["cup"] to track only cups
CAMERA_INDEX     = 0               # Index of the webcam (0 = default camera)
CAM_WIDTH        = 1280            # Capture width  (pixels)
CAM_HEIGHT       = 720             # Capture height (pixels)
DOWNSCALE_FACTOR = 2               # Speed-up factor: run detector on 1/2 size image

# ============================================================
# ------------------- YOLOv5 DETECTOR WRAPPER ----------------
# ============================================================

class YoloV5Detector:
    """
    A thin wrapper around Ultralytics YOLOv5 TorchHub model.
    Provides:
      - loading either a custom model or the standard yolov5s
      - running inference on frames
      - converting results to Deep SORT input format
    """
    def __init__(self, weights_path=None):
        # Choose GPU if available; otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # Load YOLO model
        self.model = self._load_model(weights_path)
        # YOLO keeps a dictionary/list mapping class id -> label string
        self.class_names = self.model.names

    def _load_model(self, weights_path):
        """
        Loads the YOLOv5 model.
        - If a custom weights file is provided, loads that model.
        - Otherwise loads the small pretrained yolov5s model.
        """
        if weights_path:
            model = torch.hub.load(
                'ultralytics/yolov5',      # Torch Hub repo
                'custom',                  # custom mode
                path=weights_path,         # path to weights
                force_reload=False         # avoid re-downloading every run
            )
        else:
            model = torch.hub.load(
                'ultralytics/yolov5',      # Torch Hub repo
                'yolov5s',                 # small pretrained model
                pretrained=True,
                force_reload=False
            )
        model.to(self.device).eval()
        return model

    def infer(self, frame_bgr):
        """
        Runs YOLO detection on a downscaled frame for speed.
        Returns:
            labels  – tensor of detected class ids
            boxes_n – tensor of [x1, y1, x2, y2, confidence] in NORMALIZED coords (0..1)
        """
        # Downscale input image for faster inference
        scaled_w = int(frame_bgr.shape[1] / DOWNSCALE_FACTOR)
        scaled_h = int(frame_bgr.shape[0] / DOWNSCALE_FACTOR)
        frame_small = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Run inference (YOLO handles BGR internally)
        with torch.no_grad():
            results = self.model(frame_small)

        # YOLOv5 stores outputs in results.xyxyn[0]
        labels = results.xyxyn[0][:, -1]   # last column = class id
        boxes_n = results.xyxyn[0][:, :-1] # remaining columns = [x1, y1, x2, y2, conf]
        return labels, boxes_n

    def id_to_label(self, class_id):
        """Convert a class id tensor to its human-readable label."""
        return self.class_names[int(class_id)]

    def to_deepsort_detections(self, frame_shape, infer_out,
                               conf_thresh=0.5, class_filter=None):
        """
        Convert YOLO detections to the format expected by Deep SORT.
        Args:
            frame_shape – shape of the ORIGINAL frame (H,W,3)
            infer_out   – (labels, boxes_n) from self.infer()
            conf_thresh – minimum detection confidence
            class_filter – list of class names to keep, or None for all
        Returns:
            List of tuples:
              ([x, y, w, h], confidence, class_label)
        """
        labels, boxes_n = infer_out
        dets = []
        img_h, img_w = frame_shape[:2]

        for idx in range(len(labels)):
            row = boxes_n[idx]               # [x1, y1, x2, y2, conf] normalized
            conf = float(row[4].item())
            if conf < conf_thresh:
                continue                     # skip low-confidence detections

            cls_label = self.id_to_label(labels[idx])
            if class_filter and cls_label not in class_filter:
                continue                     # skip if not in allowed class list

            # Convert normalized coords to absolute pixel coords
            x1 = int(row[0].item() * img_w)
            y1 = int(row[1].item() * img_h)
            x2 = int(row[2].item() * img_w)
            y2 = int(row[3].item() * img_h)
            w  = x2 - x1
            h  = y2 - y1
            if w <= 0 or h <= 0:
                continue                     # safety check

            tlwh = [x1, y1, w, h]            # Deep SORT expects [top-left-x, top-left-y, width, height]
            dets.append((tlwh, conf, cls_label))

        return dets

# ============================================================
# ----------------------- MAIN LOOP --------------------------
# ============================================================

def main():
    # Some environments need this to avoid OpenMP duplicate errors
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Initialise camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    # Create YOLO detector
    detector = YoloV5Detector(weights_path=CUSTOM_MODEL_PATH)

    # Create Deep SORT tracker
    # Deep SORT maintains object identities across frames using appearance embeddings.
    tracker = DeepSort(
        max_age=5,                 # max frames to keep a lost track
        n_init=2,                  # consecutive detections before confirming a track
        nms_max_overlap=1.0,       # non-max suppression overlap for merging detections
        max_cosine_distance=0.3,   # similarity threshold for re-identification
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",      # feature extractor backbone
        half=True,                 # use half precision if possible
        bgr=True,                  # input format for embedder
        embedder_gpu=torch.cuda.is_available(),  # use GPU if available
        polygon=False
    )

    print("Tracking classes:",
          "ALL" if TARGET_CLASSES is None else TARGET_CLASSES)
    print("Press ESC to quit.")

    while True:
        # Grab a frame from the camera
        ok, frame_bgr = cap.read()
        if not ok:
            print("Failed to read from camera.")
            break

        t0 = time.perf_counter()

        # 1) Run YOLO detection
        yolo_output = detector.infer(frame_bgr)

        # 2) Convert detections to Deep SORT format
        dets_for_tracker = detector.to_deepsort_detections(
            frame_shape=frame_bgr.shape,
            infer_out=yolo_output,
            conf_thresh=CONF_THRESHOLD,
            class_filter=TARGET_CLASSES
        )

        # 3) Update tracker with the current frame detections
        tracks = tracker.update_tracks(dets_for_tracker, frame=frame_bgr)

        # 4) Draw tracked objects
        for trk in tracks:
            if not trk.is_confirmed():
                continue                      # only draw confirmed tracks
            track_id = trk.track_id           # unique ID assigned by tracker
            l, t, r, b = trk.to_ltrb()        # get bounding box in left-top-right-bottom format

            # Draw rectangle around tracked object
            cv2.rectangle(frame_bgr, (int(l), int(t)), (int(r), int(b)),
                          (0, 0, 255), 2)

            # Display the unique track ID above the box
            cv2.putText(frame_bgr,
                        f"ID {track_id}",
                        (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2)

        # 5) Compute and display FPS
        dt  = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(frame_bgr,
                    f"FPS: {int(fps)}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    2)

        # 6) Show the annotated video frame
        cv2.imshow("YOLOv5 + Deep SORT Tracking", frame_bgr)

        # Exit when ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources on exit
    cap.release()
    cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    main()
