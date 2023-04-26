import cv2
import numpy as np
import glob

# Define the chessboard pattern size
pattern_size = (9, 6)

# Prepare object points (3D points in the real world)
objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

# Arrays to store object points and image points
object_points = []
image_points = []

# Read calibration images
calibration_images = glob.glob("res/calibration_images/*.jpg")

for image_path in calibration_images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        object_points.append(objp)
        image_points.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow("Chessboard", img)
        cv2.waitKey(500)
    else:
        print(f"Chessboard not detected in {image_path}")

cv2.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# Save the calibration parameters
np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
