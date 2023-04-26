import cv2
import numpy as np

def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Apply a mask to detect the ping pong ball color (white)
    _, mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    return mask

def find_ball_and_distance(frame, focal_length, real_diameter, sensor_height):
    # Preprocess the frame
    mask = preprocess_frame(frame)

    # Find contours in the frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize ball position and radius
    ball_center = None
    ball_radius = None

    for contour in contours:
        # Approximate the contour as a circle
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        # Ensure the contour is large enough
        if radius > 10:
            ball_center = (int(x), int(y))
            ball_radius = int(radius)

    # Calculate the distance to the ball if found
    distance = None
    if ball_radius:
        # Calculate the distance using the lens equation
        distance = (focal_length * real_diameter * frame.shape[0]) / (2 * ball_radius * sensor_height)

    return ball_center, ball_radius, distance

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Focal length of the camera lens (in mm)
focal_length = 3.67

# Actual diameter of the ping pong ball (in mm)
real_diameter = 40

# Height of the camera sensor (in mm)
sensor_height = 4.8

while True:
    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        break

    # Find the ball and estimate its distance
    ball_center, ball_radius, distance = find_ball_and_distance(frame, focal_length, real_diameter, sensor_height)

    if ball_center and ball_radius:
        # Draw a circle around the ball
        cv2.circle(frame, ball_center, ball_radius, (0, 255, 0), 2)

        # Display the distance
        cv2.putText(frame, f"Distance: {distance:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Ping Pong Ball Tracker", frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

