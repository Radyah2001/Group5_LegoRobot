from ultralytics import YOLO
import cv2

# model class ids
# 0: 'Back', 1: 'Ball orange', 2: 'Ball white', 3: 'Bounds', 4: 'Corner', 5: 'Cross', 6: 'Front', 7: 'Robot'
# load config
import json
with open('VideoSourceConfig.json') as f:
    config = json.load(f)

# Load a model
model = YOLO("res/best.pt")  # load a pretrained model


INPUT_SOURCE = config["InputSource"]
CONF = config["CONF"]
IOU = config["IOU"]


video = cv2.VideoCapture(INPUT_SOURCE, cv2.CAP_DSHOW)

# Loop through the video frames
while video.isOpened():
    # Read a frame from the video
    success, frame = video.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=CONF, iou=IOU)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
video.release()
cv2.destroyAllWindows()
