# app.py
import os
import glob
import io
import numpy as np
import cv2
import streamlit as st

st.set_page_config(page_title="Camera Calibration & Undistort Using OpenCV", layout="wide")


# ---------------- Utility Functions ----------------
def read_image_from_bytes(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def load_images_from_folder(folder, exts=('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG')):
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(folder, e)))
    imgs = sorted(imgs)
    out = []
    for p in imgs:
        img = cv2.imread(p)
        if img is not None:
            out.append((os.path.basename(p), img))
    return out


def auto_detect_chessboard_size(image, max_rows=12, max_cols=12, min_size=3):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for rows in range(min_size, max_rows + 1):
        for cols in range(min_size, max_cols + 1):
            found, _ = cv2.findChessboardCorners(gray, (cols, rows), None)
            if found:
                return rows, cols
    return None


def calibrate_from_images_list(images_bgr, max_search_rows=12, max_search_cols=12, min_size=3):
    if len(images_bgr) == 0:
        raise RuntimeError("No images supplied for calibration")

    first_img = next((im for _, im in images_bgr if im is not None), None)
    if first_img is None:
        raise RuntimeError("No readable images found")

    detected = auto_detect_chessboard_size(first_img, max_search_rows, max_search_cols, min_size)
    if detected is None:
        raise RuntimeError("Could not auto-detect chessboard size from first image. Try another image or increase max rows/cols.")

    nRows, nCols = detected
    objp = np.zeros((nRows * nCols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)

    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints, debug_images = [], [], []

    for name, img_bgr in images_bgr:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)
        if found and corners is not None:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            dbg = img_bgr.copy()
            cv2.drawChessboardCorners(dbg, (nCols, nRows), corners_refined, True)
            debug_images.append((name, dbg, True))
        else:
            debug_images.append((name, img_bgr.copy(), False))

    if len(objpoints) < 3:
        raise RuntimeError(f"Insufficient successful detections ({len(objpoints)}). Need >= 3 images for calibration.")

    img_shape = gray.shape[::-1]
    ret, cam_mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    return cam_mtx, dist, ret, nRows, nCols, debug_images


def undistort_image(cam_mtx, dist, img_bgr):
    h, w = img_bgr.shape[:2]
    new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, (w, h), 1, (w, h))
    und = cv2.undistort(img_bgr, cam_mtx, dist, None, new_cam_mtx)
    return und


# ---------------- UI ----------------
st.title("Camera Calibration & Undistort Using OpenCV\nBy Ubong Camilus")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_calib_imgs = st.file_uploader(
        "Upload calibration images (multi-select)", 
        type=['jpg','jpeg','png','bmp'], 
        accept_multiple_files=True
    )
    st.caption("These images will be used to compute camera calibration.")

with col2:
    calib_file = st.file_uploader("Or upload calibration .npz file", type=['npz'])
    folder_path = st.text_input("Or local folder path (when running locally)", value="")
    max_rows = st.number_input("Max search rows (inner corners)", min_value=3, max_value=30, value=12)
    max_cols = st.number_input("Max search cols (inner corners)", min_value=3, max_value=30, value=12)
    run_calib_btn = st.button("Run calibration")


# ---------------- Prepare images ----------------
images_list = []
image_names = []

if uploaded_calib_imgs:
    for up in uploaded_calib_imgs:
        data = up.read()
        img = read_image_from_bytes(data)
        if img is None:
            st.warning(f"Could not decode {up.name}")
            continue
        images_list.append((up.name, img))
        image_names.append(up.name)
elif folder_path:
    if os.path.isdir(folder_path):
        loaded = load_images_from_folder(folder_path)
        if not loaded:
            st.warning("No images found in folder using standard extensions.")
        else:
            images_list = loaded
            image_names = [n for n, _ in images_list]
    else:
        if folder_path.strip():
            st.warning("Folder path does not exist or is not accessible.")


if images_list:
    st.write(f"Loaded {len(images_list)} calibration images.")
    thumbs = []
    max_show = min(8, len(images_list))
    cols = st.columns(max_show)
    for i in range(max_show):
        nm, im = images_list[i]
        h, w = im.shape[:2]
        new_w = 160
        new_h = int(160 * h / w)
        thumb = cv2.cvtColor(cv2.resize(im, (new_w, new_h)), cv2.COLOR_BGR2RGB)
        cols[i].image(thumb, caption=nm, width='content')


# ---------------- Calibration ----------------
if run_calib_btn:
    if not images_list and calib_file is None:
        st.error("No images or calibration file supplied.")
    else:
        try:
            if images_list:
                with st.spinner("Running auto-detection and calibration..."):
                    cam_mtx, dist_coeffs, reproj_err, nRows, nCols, debug_imgs = calibrate_from_images_list(
                        images_list, max_search_rows=max_rows, max_search_cols=max_cols, min_size=3
                    )
                st.success("Calibration finished")
            elif calib_file:
                npz_data = np.load(calib_file)
                cam_mtx = npz_data['cam_matrix']
                dist_coeffs = npz_data['dist_coeffs']
                reproj_err = npz_data.get('reprojection_error', None)
                st.success("Calibration file loaded successfully")

            # Save calibration to session state
            st.session_state['cam_mtx'] = cam_mtx
            st.session_state['dist_coeffs'] = dist_coeffs
            st.session_state['reproj_err'] = reproj_err

            # Show calibration results
            st.markdown("**Camera matrix:**")
            st.code(np.array2string(cam_mtx, precision=5, separator=', '))
            st.markdown("**Distortion coefficients:**")
            st.code(np.array2string(dist_coeffs.ravel(), precision=6, separator=', '))
            if reproj_err is not None:
                st.write(f"**Reprojection error (RMS):** {reproj_err:.6f}")

            # Save .npz
            save_path = os.path.join(os.getcwd(), "calibration.npz")
            np.savez(save_path, reprojection_error=reproj_err, cam_matrix=cam_mtx, dist_coeffs=dist_coeffs)
            with open(save_path, "rb") as f:
                calib_bytes = f.read()
            st.download_button("Download calibration.npz", data=calib_bytes, file_name="calibration.npz")

            # Show debug images if computed
            if images_list:
                st.markdown("### Debug: detection results (green = OK)")
                debug_cols = st.columns(3)
                for i, (name, dbg_img, ok) in enumerate(debug_imgs[:9]):
                    rgb = cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB)
                    caption = f"{name} - {'OK' if ok else 'FAIL'}"
                    debug_cols[i % 3].image(rgb, caption=caption, width='stretch')

        except Exception as e:
            st.error(f"Calibration failed: {e}")
            st.exception(e)


# ---------------- Undistortion ----------------
st.markdown("---")
st.markdown("### Undistort image(s)")

undistort_upload = st.file_uploader(
    "Upload image(s) to undistort", 
    type=['jpg','jpeg','png','bmp'], 
    accept_multiple_files=True
)
undistort_btn = st.button("Run undistortion")

if undistort_btn:
    if 'cam_mtx' not in st.session_state or 'dist_coeffs' not in st.session_state:
        st.warning("You must run calibration or load a .npz file first.")
    elif not undistort_upload:
        st.warning("Please upload at least one image to undistort.")
    else:
        cam_mtx = st.session_state['cam_mtx']
        dist_coeffs = st.session_state['dist_coeffs']
        for up in undistort_upload:
            data = up.read()
            img = read_image_from_bytes(data)
            if img is None:
                st.warning(f"Could not decode {up.name}")
                continue
            und_img = undistort_image(cam_mtx, dist_coeffs, img)
            st.image(
                [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(und_img, cv2.COLOR_BGR2RGB)],
                caption=[f"{up.name} — Original", f"{up.name} — Undistorted"],
                width=400
            )


# ---------------- Sidebar Notes ----------------
st.sidebar.markdown("## Notes")
st.sidebar.write("""
- Upload at least 6–10 good chessboard images for calibration.
- Ensure chessboard is fully visible and not blurred.
- If auto-detection fails, increase 'Max search rows/cols'.
""")
