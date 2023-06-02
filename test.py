from roboflow import Roboflow
import cv2

rf = Roboflow(api_key="cl34nreOJIaV0Dj3jYng")
project = rf.workspace().project("ping-pong-finder-w6mxk")
model = project.version(9).model

model.webcam(webcam_id=1)
