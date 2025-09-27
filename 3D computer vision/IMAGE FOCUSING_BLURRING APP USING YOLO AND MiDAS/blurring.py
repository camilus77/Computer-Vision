import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO


IMAGE_PATH = "/content/soccer.jpg"  #image address
DEPTH_MODEL_TYPE = "DPT_Hybrid"      # "DPT_Large" (highest quality), "DPT_Hybrid" (good), "MiDaS_small" (fast)
SEG_MODEL_WEIGHTS = "yolov8n-seg.pt" # lightweight; alternatives: yolov8s-seg.pt, yolov8m-seg.pt


def require(cond, msg):
    # Small assertion helper: raises a ValueError with a custom message when a condition fails.
    if not cond:
        raise ValueError(msg)

def show_img(img, title=None, cmap=None, size=(8,5)):
    # Convenience function to display an image with matplotlib.
    # - size controls figure size
    # - cmap is optional (used for grayscale/heatmaps)
    # - title is optional
    plt.figure(figsize=size)
    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    if title: plt.title(title)
    plt.axis("off")


def to_uint8_mask(mask_bool_or_float):
    # Ensures a mask is uint8 in the 0..255 range (OpenCV-friendly).
    # If mask is binary 0/1, expand to 0/255. If already uint8 0..255, leave as is.
    m = mask_bool_or_float.astype(np.uint8)
    if m.max() == 1:  # binary 0/1 -> 0/255
        m = m * 255
    return m


# Validate that the IMAGE exists on disk.
require(os.path.exists(IMAGE_PATH), f"Image not found: {IMAGE_PATH}")
# Read the image in BGR format (OpenCV default).
bgr = cv2.imread(IMAGE_PATH)
# Ensure the read succeeded (cv2 returns None on failure).
require(bgr is not None, f"cv2.imread failed for {IMAGE_PATH}")
# Convert to RGB for consistent plotting with matplotlib and model expectations.
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
# Capture original image height and width for later resizing/consistency.
H, W = rgb.shape[:2]

# =========================
#Monocular depth with MiDaS
# =========================
# Pick GPU if available; otherwise, fall back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load MiDaS depth model from PyTorch Hub based on DEPTH_MODEL_TYPE.
# .eval() puts model in inference mode; .to(device) moves it to GPU/CPU.
midas = torch.hub.load("intel-isl/MiDaS", DEPTH_MODEL_TYPE).to(device).eval()
# Load the corresponding preprocessing transforms appropriate for MiDaS variants.
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Choose the correct transform function for the specified model.
if DEPTH_MODEL_TYPE in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Convert PIL Image to numpy array before applying the transform
# (The chosen MiDaS transform expects a PIL Image; here we ensure proper type/format).
input_batch = transform(np.array(Image.fromarray(rgb))).to(device)


with torch.no_grad():
    # Forward pass through MiDaS to get a single-channel depth prediction.
    prediction = midas(input_batch)
    # Resize the predicted depth map back to the original image dimensions (H, W).
    # MiDaS outputs are often smaller; we use bicubic interpolation for smooth resizing.
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),   # add channel dim: (B,1,H,W)
        size=(H, W),               # target spatial size
        mode="bicubic",
        align_corners=False,
    ).squeeze()                    # remove the added channel dim to get (H, W)

# Move depth from GPU tensor to CPU numpy for OpenCV/numpy processing.
depth = prediction.cpu().numpy()

# Normalize depth to 0..255 (higher = closer)
# MiDaS is inverse depth-like; after normalize, bright ~ closer (good for our logic)
# cv2.normalize maps min->0 and max->255 linearly.
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
# Convert to uint8 for visualization and OpenCV operations.
depth_u8 = depth_norm.astype(np.uint8)

# Visualize the depth map using a perceptual colormap; bright = closer.
show_img(depth_u8, "Depth Map (255 = closer)", cmap="inferno", size=(6,4))
plt.colorbar(); plt.show()

# =========================
# 4) Instance segmentation with YOLOv8-Seg
# =========================
# Load an Ultralytics YOLOv8 segmentation model (pretrained weights).
seg_model = YOLO(SEG_MODEL_WEIGHTS)
# Run inference on the RGB image. imgsz sets the inference size; conf sets confidence threshold.
# Lower conf => more detections; higher conf => fewer, more confident detections.
results = seg_model.predict(source=rgb, imgsz=max(H, W), conf=0.25, verbose=False)

# Ultralytics returns a list (one element per image). Ensure we got at least one result.
require(len(results) > 0, "No YOLO results.")
res = results[0]

# Ensure instance masks are present in the result. If not, advise switching to a larger model or different image.
require(res.masks is not None and res.masks.data is not None and len(res.masks.data) > 0,
        "No instance masks detected. Try a larger model (e.g., yolov8s-seg.pt) or another image.")

# res.masks.data is a tensor of shape [N, Hmask, Wmask], values ~{0,1} per instance.
masks = res.masks.data.cpu().numpy()  # shape: [N, Hm, Wm]
# If YOLO resized internally, masks are often aligned to its internal inference size.
# We ensure masks match the original image size (H, W) for pixelwise operations with depth_u8.
if masks.shape[1] != H or masks.shape[2] != W:
    # Resize each mask to (H, W) using nearest neighbor to keep binary nature.
    resized_masks = []
    for m in masks:
        m_u8 = (m * 255).astype(np.uint8)                    # scale to 0..255 for OpenCV resize
        m_rs = cv2.resize(m_u8, (W, H), interpolation=cv2.INTER_NEAREST)
        resized_masks.append((m_rs > 127).astype(np.uint8))  # threshold back to 0/1
    masks = np.stack(resized_masks, axis=0)
else:
    # If already the right size, just binarize at 0.5.
    masks = (masks > 0.5).astype(np.uint8)

# Visualize all-object mask
# Combine all instance masks into a single mask (any pixel covered by any instance set to 255).
all_mask = (masks.sum(axis=0) > 0).astype(np.uint8) * 255
show_img(all_mask, "All Segmented Objects", cmap="gray", size=(6,4))
plt.show() # Add plt.show() for this plot

# =========================
#CHECK OBJECT CLOSEST TO THE CAMERA
# =========================
# Strategy: For each instance mask, compute the mean depth value under that mask
# and select the one with the highest average (i.e., closest based on normalized inverse depth).
closest_idx = None
max_avg_depth = -1.0

for i in range(masks.shape[0]):
    m = masks[i]                 # binary mask 0/1 for instance i
    vals = depth_u8[m > 0]       # depth values inside the instance
    if vals.size == 0:
        continue
    avg = float(np.mean(vals))   # average depth within the mask region
    if avg > max_avg_depth:      # keep the instance with the highest average (closest)
        max_avg_depth = avg
        closest_idx = i

# Ensure we found at least one valid instance as "closest".
require(closest_idx is not None, "Could not determine closest instance.")

# Extract that instance’s mask and cast it to a 0/255 uint8 mask suitable for OpenCV ops.
closest_mask = masks[closest_idx]  # 0/1
closest_mask_u8 = to_uint8_mask(closest_mask)

# ISOLATE MASK OF CLOSEST OBJECT

# =========================
# To reduce bleeding and better isolate the object, we intersect the instance mask
# with a depth "band" that covers pixels whose depth is within [min-10, max+10] of the object.
obj_depth_vals = depth_u8[closest_mask > 0]
require(obj_depth_vals.size > 0, "Empty depth values for selected object.")
mn, mx = int(obj_depth_vals.min()), int(obj_depth_vals.max())
pad = 10
lower, upper = max(mn - pad, 0), min(mx + pad, 255)

# Build a depth range mask, then AND with the instance mask to keep only plausible depths.
depth_band = cv2.inRange(depth_u8, lower, upper)
final_mask = cv2.bitwise_and(closest_mask_u8, depth_band)
final_mask = to_uint8_mask(final_mask)

# Visualize the refined mask of the nearest object.
show_img(final_mask, "Final Focus Mask (Closest Object)", cmap="gray", size=(6,4))
plt.show() # Add plt.show() for this plot



#BLUR OBJECT
# Create a background-blurred version of the original image (portrait-mode effect).
blurred = cv2.GaussianBlur(rgb, (21, 21), 0)
# Build a 3-channel version of the final single-channel mask to apply on RGB images.
mask_3 = np.dstack([final_mask]*3)
# Invert the mask so background is selected where mask == 0.
inv_3  = cv2.bitwise_not(mask_3)

# Extract sharp foreground (object) and blurred background, then add them together.
fg = cv2.bitwise_and(rgb, mask_3)
bg = cv2.bitwise_and(blurred, inv_3)
final = cv2.add(fg, bg)

# Show the final composited image: closest object remains sharp, surroundings are blurred.
show_img(final, "Final Output: Closest Object Focused", size=(10,6))
plt.show() # Add plt.show() for this plot
