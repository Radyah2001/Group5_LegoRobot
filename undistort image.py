import cv2
import numpy as np

# Load the camera calibration parameters
with np.load("camera_calibration.npz") as calibration_file:
    camera_matrix = calibration_file["camera_matrix"]
    dist_coeffs = calibration_file["dist_coeffs"]

# Initialize the webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Webcam not found.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Frame not captured.")
        break

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Display the undistorted frame
    cv2.imshow("Undistorted Frame", undistorted_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
