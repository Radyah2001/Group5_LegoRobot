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
import torchvision

# Change format from image to json
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json", # Change to json to get prediction boxes
    "&stroke=5"
])


# Get webcam interface via opencv-python
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# The size of the cells in the grid.
cell_size = 50  # This value can be adjusted. About 50 seems to be a good fit for now.

# This function maps a bounding box in pixel coordinates to a grid system. For each bounding box, it calculates which
# grid cells it intersects with and returns a list of those grid cell coordinates.
def map_to_grid(img_height, img_width, cell_size, bounding_box):
    # Unpack the bounding box coordinates
    x, y, width, height = bounding_box

    # Calculate the total number of cells in the grid along the width and height of the image
    grid_width = img_width // cell_size
    grid_height = img_height // cell_size

    # Calculate the grid cell coordinates for the top-left corner of the bounding box
    # We achieve this by dividing the pixel coordinates by the cell size
    start_x = x // cell_size
    start_y = y // cell_size

    # Similarly, calculate the grid cell coordinates for the bottom-right corner of the bounding box
    end_x = (x + width) // cell_size
    end_y = (y + height) // cell_size

    # We must ensure that the calculated grid coordinates don't exceed the grid's boundaries
    # If they do, we cap them at the maximum allowed coordinate which is grid size - 1
    start_x = max(0, min(grid_width - 1, start_x))
    start_y = max(0, min(grid_height - 1, start_y))
    end_x = max(0, min(grid_width - 1, end_x))
    end_y = max(0, min(grid_height - 1, end_y))

    # We will store all the grid cells that intersect with the bounding box in this list
    cells = []

    # For each grid cell in the range calculated above, we add the cell to our list
    # Note: The +1 in the range is because the end index is exclusive in Python ranges
    for i in range(start_x, end_x + 1):
        for j in range(start_y, end_y + 1):
            cells.append((i, j))

    # Finally, we return the list of grid cells that intersect with the bounding box
    return cells



# Infer via the Roboflow Infer API and return the result
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

    # Parse JSON result
    json_data = resp.json()

    # Now `json_data` contains the data from the response, including bounding box information
    # The exact structure of the JSON data will depend on the API, but it will generally include
    # the class, score, and bounding box coordinates for each detected object.

    for prediction in json_data['predictions']:
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        label = prediction['class']
        score = prediction['confidence']

        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # map the bounding box to the grid
        cells = map_to_grid(height, width, cell_size, (x, y, width, height))

        # Write the grid cell coordinates on the image
        cell_text = ', '.join(map(str, cells))
        cv2.putText(img, cell_text, (x, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"{label} is in grid cells: {cells}")

    # Draw grid on the inference result image
    height, width, _ = img.shape
    for y in range(0, height, cell_size):
        cv2.line(img, (0, y), (width, y), (0, 0, 255), 1)
    for x in range(0, width, cell_size):
        cv2.line(img, (x, 0), (x, height), (0, 0, 255), 1)

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
            image = await futures.pop(0)
            # And display the inference results
            cv2.imshow('image', image)

# Run our main loop
asyncio.run(main())

# Release resources when finished
video.release()
cv2.destroyAllWindows()