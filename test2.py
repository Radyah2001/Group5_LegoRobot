import cv2
import numpy as np

# Create a video capture object for the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera was successfully opened
if not cap.isOpened():
    print("Failed to open camera")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame and filter for white objects
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]
            mean_color = cv2.mean(roi)
            if mean_color[0] > 200 and mean_color[1] > 200 and mean_color[2] > 200:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Display the original frame with white object detection
    cv2.imshow('White Object Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
