# load config
import json
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

import asyncio
import cv2
import base64
import numpy as np
import httpx
import time

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json",  # Change from image to json
    "&stroke=5"
])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# This function maps a point in the image to a point in the 2D coordinate system
def map_point(point, homography_matrix):
    """
    point: a tuple (x, y) representing the point in the image
    homography_matrix: the homography matrix obtained from cv2.findHomography
    """
    point = np.array([[point]], dtype='float32')
    mapped_point = cv2.perspectiveTransform(point, homography_matrix)
    return (mapped_point[0][0][0], mapped_point[0][0][1])

# These are the corners of the course in the image, obtained using your object detection model
# You need to replace these with the actual values
image_points = np.float32([
    [x1, y1],  # top-left corner
    [x2, y2],  # top-right corner
    [x3, y3],  # bottom-right corner
    [x4, y4],  # bottom-left corner
])

# These are the corners of the course in the 2D coordinate system
course_points = np.float32([
    [0, 0],             # top-left corner (0, 0)
    [180, 0],           # top-right corner (180, 0)
    [180, 120],         # bottom-right corner (180, 120)
    [0, 120],           # bottom-left corner (0, 120)
])

# Compute the homography matrix
H, mask = cv2.findHomography(image_points, course_points)

# Now you can map any point in the image to a point in the 2D coordinate system
# For example, to map the center of the robot (replace with actual values)
robot_center_image = (x_robot, y_robot)
robot_center_course = map_point(robot_center_image, H)


# Infer via the Roboflow Infer API and return the result
# Takes an httpx.AsyncClient as a parameter
async def infer(requests):
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
    resp = await requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })

    # Parse result into JSON
    data = resp.json()

    # Extract detections
    detections = data['predictions']

    corners = []
    robot_center = None

    for detection in detections:
        class_name = detection['class']
        box = detection['bbox']
        center = ((box['left'] + box['width'] / 2), (box['top'] + box['height'] / 2))

        if class_name == 'Corner':
            corners.append(center)
        elif class_name == 'Robot':
            robot_center = center

    # Sort corners based on their coordinates to match the ordering in course_points
    corners = sorted(corners, key=lambda point: (-point[1], point[0]))

    # Now corners contains the center points of the corners and robot_center is the center point of the robot

    return img

# Main loop; infers at FRAMERATE frames per second until you press "q"
async def main():
    # Initialize
    last_frame = time.time()

    # Initialize a buffer of images
    futures = []

    async with httpx.AsyncClient() as requests:
        while 1:
            # On "q" keypress, exit
            if(cv2.waitKey(1) == ord('q')):
                break

            # Throttle to FRAMERATE fps and print actual frames per second achieved
            elapsed = time.time() - last_frame
            await asyncio.sleep(max(0, 1/FRAMERATE - elapsed))
            print((1/(time.time()-last_frame)), " fps")
            last_frame = time.time()

            # Enqueue the inference request and safe it to our buffer
            task = asyncio.create_task(infer(requests))
            futures.append(task)

            # Wait until our buffer is big enough before we start displaying results
            if len(futures) < BUFFER * FRAMERATE:
                continue

            # Remove the first image from our buffer
            # wait for it to finish loading (if necessary)
            image = await futures.pop(0)
            # And display the inference results
            cv2.imshow('image', image)

# Run our main loop
asyncio.run(main())

# Release resources when finished
video.release()
cv2.destroyAllWindows()