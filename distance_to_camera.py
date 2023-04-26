import cv2
import numpy as np

# Dimensions of A4 paper in centimeters
a4_width = 21.0
a4_height = 29.7

# Initialize the camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Define the font and color for displaying the distance
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 255)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours in the edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    max_contour = contours[max_index]

    # Find the corners of the paper
    perimeter = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * perimeter, True)

    # If four corners are detected, calculate the distance
    if len(approx) == 4:
        # Draw the corners on the frame
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        # Calculate the width and height of the paper in pixels
        width = np.linalg.norm(approx[0] - approx[1])
        height = np.linalg.norm(approx[1] - approx[2])

        # Calculate the distance to the paper
        if width > height:
            distance = (a4_width * 0.5) / (width / frame.shape[1])
        else:
            distance = (a4_height * 0.5) / (height / frame.shape[0])

        # Display the distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30), font, font_scale, font_color, 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("frame", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close the window
cap.release()
out.release()
cv2.destroyAllWindows()
