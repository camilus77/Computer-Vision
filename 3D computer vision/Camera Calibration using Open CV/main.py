import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt


def auto_detect_chessboard_size(image, max_rows=12, max_cols=12):
    """
    Automatically tries different (rows, cols) to detect chessboard pattern.
    Returns the first valid (rows, cols) that works, else None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for rows in range(3, max_rows + 1):
        for cols in range(3, max_cols + 1):
            ret, _ = cv2.findChessboardCorners(gray, (cols, rows), None)
            if ret:
                return rows, cols
    return None


def calibrate(showPics=True):
    """
    Calibrate a single camera using chessboard images.
    Automatically detects chessboard rows and cols.
    Returns: cam_matrix, dist_coeffs
    Saves calibration parameters to 'calibration.npz'.
    """
    calibration_dir = r"C:\\Users\\CLINTON\\Desktop\\Ubong python\\Camera Calibration using Open CV"
    img_path_list = glob.glob(os.path.join(calibration_dir, '*.jpg'))

    if not img_path_list:
        raise RuntimeError(f"No .jpg images found in {calibration_dir}")

    # load first image for auto-detect
    first_img = cv2.imread(img_path_list[0])
    if first_img is None:
        raise RuntimeError("First image could not be read for chessboard detection")

    detected_size = auto_detect_chessboard_size(first_img)
    if detected_size is None:
        raise RuntimeError("No chessboard pattern detected in first image. Check your chessboard images.")

    nRows, nCols = detected_size
    print(f"Auto-detected chessboard inner corners: {nRows} rows × {nCols} cols")

    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points for one pattern view
    world_pts_single = np.zeros((nRows * nCols, 3), np.float32)
    world_pts_single[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)

    world_pts_list = []
    img_pts_list = []

    for cur_img_path in img_path_list:
        img_bgr = cv2.imread(cur_img_path)
        if img_bgr is None:
            print(f"Warning: Could not read {cur_img_path}, skipping.")
            continue

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        corners_found, corners_org = cv2.findChessboardCorners(img_gray, (nRows, nCols), None)

        if corners_found:
            world_pts_list.append(world_pts_single)
            corners_refined = cv2.cornerSubPix(img_gray, corners_org, (11, 11), (-1, -1), term_criteria)
            img_pts_list.append(corners_refined)

            if showPics:
                cv2.drawChessboardCorners(img_bgr, (nRows, nCols), corners_refined, corners_found)
                cv2.imshow('Chessboard', img_bgr)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    if len(world_pts_list) == 0 or len(img_pts_list) == 0:
        raise RuntimeError("No chessboard corners were found in any image set. Check your dataset.")

    # Calibrate camera
    ret, cam_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        world_pts_list, img_pts_list, img_gray.shape[::-1], None, None
    )
    print("Camera Matrix:\n", cam_matrix)
    print("Reproj Error (pixels): {:.4f}".format(ret))

    # Save params
    cur_folder = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    param_path = os.path.join(cur_folder, 'calibration.npz')
    np.savez(param_path,
             reprojection_error=ret,
             cam_matrix=cam_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs)
    print(f"Saved calibration parameters to: {param_path}")

    return cam_matrix, dist_coeffs


def remove_distortion(cam_matrix, dist_coeffs, example_img_path=None):
    """
    Read an example image and undistort it using the given camera matrix and distortion coefficients.
    """
    if example_img_path is None:
        example_img_path = 'C:\\Users\\CLINTON\\Desktop\\Ubong python\\Camera Calibration using Open CV\\chess images\\leftcamera\\Im_L_4.png'

    img = cv2.imread(example_img_path)
    if img is None:
        raise RuntimeError(f"Could not read example image at {example_img_path}")

    height, width = img.shape[:2]
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (width, height), 1, (width, height))
    img_undist = cv2.undistort(img, cam_matrix, dist_coeffs, None, new_cam_matrix)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_undist_rgb = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Undistorted')
    plt.imshow(img_undist_rgb)
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    cam_mtx, dist = calibrate(showPics=True)
    remove_distortion(cam_mtx, dist)
