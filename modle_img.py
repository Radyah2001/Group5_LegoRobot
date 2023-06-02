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
    "&format=image",
    "&stroke=5"
])

# Load static image using cv2.imread()
img = cv2.imread('test.jpg')

# Infer via the Roboflow Infer API and return the result
# Takes an httpx.AsyncClient as a parameter
async def infer(requests):
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

    # Parse result image
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

# Main function to run the infer function
async def main():
    async with httpx.AsyncClient() as requests:
        image = await infer(requests)
        # Display the inference results
        cv2.imshow('image', image)
        cv2.waitKey(0)

# Run our main function
asyncio.run(main())

cv2.destroyAllWindows()
