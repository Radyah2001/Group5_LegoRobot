import json
import cv2
import base64
import numpy as np
import httpx
import asyncio

# load config
with open('roboflow_config.json') as f:
    config = json.load(f)

ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

# Construct the Roboflow Infer URL
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&stroke=5"
])

def get_center(x, y, width, height):
    x_center = x + (width / 2)
    y_center = y + (height / 2)
    return (x_center, y_center)


async def infer(requests, img):
    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img_resized = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img_resized)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = await requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })

    # Print out the content of the response to understand its structure
    print(resp.json())

    # Parse result image
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

async def main():
    async with httpx.AsyncClient() as requests:
        detections, img = await infer(requests)

        # Get corners and balls locations
        corners = [get_center(det['x'], det['y'], det['width'], det['height'])
                   for det in detections if det['class'] == 'Corner']
        ping_pong_balls = [get_center(det['x'], det['y'], det['width'], det['height'])
                           for det in detections if det['class'] == 'Ball white']

        # Check if we have all corners
        if len(corners) != 4:
            raise ValueError('We should detect 4 corners')

        # Four corners of the field in the image (pixel coordinates)
        src_pts = np.float32(corners)

        # Four corners of the field in real-world coordinates (cm)
        dst_pts = np.float32([[0, 0], [180, 0], [180, 120], [0, 120]])

        # Compute the homography
        H, _ = cv2.findHomography(src_pts, dst_pts)

        real_world_ping_pong_balls = []
        for (px, py) in ping_pong_balls:
            # Homogeneous coordinates for the pixel position
            pixel_pos = np.float32([px, py, 1]).reshape(-1, 1)

            # Multiply by the homography
            real_world_pos = np.dot(H, pixel_pos)

            # Convert back to non-homogeneous coordinates
            real_world_pos /= real_world_pos[2, 0]

            # Append to the list of real world positions
            real_world_ping_pong_balls.append((real_world_pos[0, 0], real_world_pos[1, 0]))

        # Now real_world_ping_pong_balls contains the real-world coordinates of the ping pong balls
        print(real_world_ping_pong_balls)

        # Display the inference results
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Run our main function
asyncio.run(main())
