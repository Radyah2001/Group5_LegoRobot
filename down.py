import json
import cv2
import base64
import math
import numpy as np
import requests
import time

# load config
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Construct the Roboflow Infer URL
# obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json",  # Change to json if you want the prediction boxes, not the visualization
    "&stroke=5"
])

# Get webcam interface via opencv-python
# Replace with path to video file
video = cv2.VideoCapture(1, cv2.CAP_DSHOW)


# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True)

    predictions = resp.json()
    detections = predictions['predictions']

    robot_center = None
    arrow_center = None
    angle_deg = None
    color = (0, 0, 255)
    # Variables to store the closest ball and its distance to the robot
    closest_ball = None
    closest_ball_distance = float('inf')

    # Parse result image and calculate angle
    for bounding_box in detections:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        class_name = bounding_box['class']
        confidence = bounding_box['confidence']

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        # draw/place bounding boxes on image
        cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)

        (text_width, text_height), _ = cv2.getTextSize(
            f"{class_name} | {confidence}",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)

        cv2.rectangle(img, (int(x0), int(y0)), (int(x0) + text_width, int(y0) - text_height), color=(0,0,0),
                      thickness=-1)

        text_location = (int(x0), int(y0))

        cv2.putText(img, f"{class_name} | {confidence}",
                    text_location, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                    color=(255,255,255), thickness=2)

        if class_name == 'Robot':
            robot_center = (center_x, center_y)
            color = (0, 255, 0)  # Green for robot

        elif class_name == 'Front':
            arrow_center = (center_x, center_y)
            color = (0, 0, 255)  # Red for front

        elif class_name in ['Ball white', 'Ball orange']:
            # Calculate the Euclidean distance between the robot and the ball
            if robot_center is not None:
                distance = math.sqrt((center_x - robot_center[0])**2 + (center_y - robot_center[1])**2)
                # If this ball is closer than the current closest ball, update the closest ball and its distance
                if distance < closest_ball_distance:
                    closest_ball = (center_x, center_y)
                    closest_ball_distance = distance

    if robot_center and arrow_center:
        # calculate differences in x and y coordinates
        diff_x = arrow_center[0] - robot_center[0]
        diff_y = arrow_center[1] - robot_center[1]

        # calculate angle in radians
        angle_rad = math.atan2(-diff_y, diff_x)

        # convert angle to degrees
        angle_deg = (math.degrees(angle_rad) + 360) % 360

        print("Robot is facing at angle:", angle_deg, "degrees")

    # Print the coordinates of the closest ball, if any
    if closest_ball is not None:
        print("Closest ball is at coordinates:", closest_ball)

    return img, detections, angle_deg


# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if (cv2.waitKey(1) == ord('q')):
        break

    # Capture start time to calculate fps
    start = time.time()

    # Synchronously get a prediction from the Roboflow Infer API
    image, detections, angle_deg = infer()
    # And display the inference results
    cv2.imshow('image', image)

    # Print frames per second
    print((1 / (time.time() - start)), " fps")
    print(detections)
    print(angle_deg)

# Release resources when finished
video.release()
cv2.destroyAllWindows()
