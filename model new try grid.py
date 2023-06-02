import numpy as np
import cv2
import asyncio
import httpx
import time
import json
import base64

from sympy.parsing.sympy_parser import TOKEN

# Load configuration
with open('roboflow_config.json') as f:
    config = json.load(f)

ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

FRAMERATE = config["FRAMERATE"]
BUFFER = config["BUFFER"]

upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json",
    "&stroke=5"
])

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Define grid parameters
GRID_SIZE = 10  # Each grid cell is 10x10 cm
GRID_WIDTH = 20  # 180 cm / 10 cm/grid cell
GRID_HEIGHT = 14  # 120 cm / 10 cm/grid cell

# Define a grid to hold our objects
grid = [['empty' for _ in range(GRID_WIDTH)] for __ in range(GRID_HEIGHT)]

def parse_objects(predictions):
    global grid

    for obj in predictions:
        x = min(max(0, round(obj['x'] / GRID_SIZE)), GRID_SIZE - 1)
        y = min(max(0, round(obj['y'] / GRID_SIZE)), GRID_SIZE - 1)

        if grid[y][x] == 'empty':
            grid[y][x] = obj['class']

    return grid

async def infer(requests):
    ret, img = video.read()

    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Add a grid to the image
    img_grid = img.copy()
    cell_width = img_grid.shape[1] // GRID_WIDTH
    cell_height = img_grid.shape[0] // GRID_HEIGHT

    for i in range(GRID_HEIGHT + 1):
        cv2.line(img_grid, (0, i * cell_height), (img_grid.shape[1], i * cell_height), (255, 255, 255), 1)

    for j in range(GRID_WIDTH + 1):
        cv2.line(img_grid, (j * cell_width, 0), (j * cell_width, img_grid.shape[0]), (255, 255, 255), 1)

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img_grid)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    try:
        resp = await requests.post(upload_url, data=img_str, headers={
            "Content-Type": "application/octet-stream",
            "Authorization": f"Bearer {TOKEN}"
        }, timeout=30.0)
    except RuntimeError as e:
        print(f"An error occurred during the post request: {str(e)}")
        return None


    # Parse the JSON response
    result = json.loads(resp.content)

    # Update the grid with the detected objects
    parse_objects(result['predictions'])

    # Overlay the grid onto the image
    cell_height = img.shape[0] // GRID_HEIGHT
    cell_width = img.shape[1] // GRID_WIDTH
    for i in range(0, img.shape[1], cell_width):
        cv2.line(img, (i, 0), (i, img.shape[0]), color=(255, 255, 255), thickness=1)
    for i in range(0, img.shape[0], cell_height):
        cv2.line(img, (0, i), (img.shape[1], i), color=(255, 255, 255), thickness=1)

    # Show the image with the grid
    cv2.imshow('image', img)

    return result['predictions']

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
            predictions = await futures.pop(0)

            # Print the grid to the console
            for row in grid:
                print(' '.join(row))

asyncio.run(main())
video.release()
cv2.destroyAllWindows()
