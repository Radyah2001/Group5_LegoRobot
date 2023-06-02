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
    "&format=json",
    "&stroke=5"
])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Parameters of your setup
FIELD_WIDTH_CM = 120
FIELD_LENGTH_CM = 180
GRID_CELL_SIZE_CM = 10
WEBCAM_HEIGHT_PX = 1080
WEBCAM_WIDTH_PX = 1920  # Assuming a 16:9 aspect ratio for a 1080p image

# Calculate the scaling factor from pixels to centimeters
PX_TO_CM_HEIGHT = FIELD_LENGTH_CM / WEBCAM_HEIGHT_PX
PX_TO_CM_WIDTH = FIELD_WIDTH_CM / WEBCAM_WIDTH_PX

# Calculate the number of cells in the grid
GRID_WIDTH_CELLS = int(FIELD_WIDTH_CM / GRID_CELL_SIZE_CM)
GRID_LENGTH_CELLS = int(FIELD_LENGTH_CM / GRID_CELL_SIZE_CM)

# Infer via the Roboflow Infer API and return the result
# Takes an httpx.AsyncClient as a parameter
async def infer(requests):
    # Get the current image from the webcam
    ret, frame = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = frame.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(frame, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = await requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, timeout=30.0)  # 30 second timeout

    # Extract JSON response from Roboflow API
    response_json = resp.json()

    # Map each detected object to a grid cell
    for prediction in response_json['predictions']:
        # Extract the object's center point in pixels
        x_px = prediction['x'] * WEBCAM_WIDTH_PX
        y_px = prediction['y'] * WEBCAM_HEIGHT_PX

        # Convert to real-world dimensions in cm
        x_cm = x_px * PX_TO_CM_WIDTH
        y_cm = y_px * PX_TO_CM_HEIGHT

        # Map to grid cell
        grid_x = int(x_cm / GRID_CELL_SIZE_CM)
        grid_y = int(y_cm / GRID_CELL_SIZE_CM)

        print(f"Object {prediction['class']} is in grid cell ({grid_x}, {grid_y})")

        # Draw bounding box on the image
        # Calculate top-left point of the bounding box
        tl_x = int(x_px - prediction['width'] * WEBCAM_WIDTH_PX / 2)
        tl_y = int(y_px - prediction['height'] * WEBCAM_HEIGHT_PX / 2)

        # Calculate bottom-right point of the bounding box
        br_x = int(x_px + prediction['width'] * WEBCAM_WIDTH_PX / 2)
        br_y = int(y_px + prediction['height'] * WEBCAM_HEIGHT_PX / 2)

        # Draw bounding box
        cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (0, 255, 0), 2)

    # Show the image
    cv2.imshow('image', frame)


# Main loop; infers at FRAMERATE frames per second until you press "q"
async def main():
    # Initialize
    last_frame = time.time()

    # Initialize a buffer of images
    futures = []

    async with httpx.AsyncClient() as requests:
        while 1:
            # On "q" keypress, exit
            if (cv2.waitKey(1) == ord('q')):
                break

            # Throttle to FRAMERATE fps and print actual frames per second achieved
            elapsed = time.time() - last_frame
            await asyncio.sleep(max(0, 1 / FRAMERATE - elapsed))
            print((1 / (time.time() - last_frame)), " fps")
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
            # cv2.imshow('image', image)


# Run our main loop
asyncio.run(main())

# Release resources when finished
video.release()
cv2.destroyAllWindows()