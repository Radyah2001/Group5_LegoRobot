import cv2
import numpy as np

def preprocess_frame(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for the ping pong ball (you can adjust these values based on the ball color)
    lower_color = np.array([20, 100, 100]) # Adjust based on ball color
    upper_color = np.array([40, 255, 255]) # Adjust based on ball color

    # Apply the color mask
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask

def is_circular(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return 0.7 < circularity < 1.2

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