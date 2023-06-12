import socket

import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np
import math
import json

with open('VideoSourceConfig.json') as f:
    config = json.load(f)

CONF = config["CONF"]
IOU = config["IOU"]
INPUT_SOURCE = config["InputSource"]

host = "192.168.43.168"  # get local machine name
port = 1060  # Make sure it's within the > 1024 $$ <65535 range

s = socket.socket()
s.connect((host, port))

def turn_robot(robot_angle, robot_coord, ball_coord, isMoving):
    # calculate differences in x and y coordinates
    diff_x = ball_coord[0] - robot_coord[0]
    diff_y = ball_coord[1] - robot_coord[1]

    # calculate angle in radians
    angle_rad = math.atan2(-diff_y, diff_x)

    # calculate distance
    dist = robot_coord[0] - ball_coord[0]

    # convert angle to degrees
    target_angle = (math.degrees(angle_rad) + 360) % 360

    angle_difference = target_angle - robot_angle
    if angle_difference > 180:
        angle_difference -= 360
    elif angle_difference < -180:
        angle_difference += 360

    if -5 <= angle_difference <= 5:  # Close enough to target
        message = "STOP"
        s.send(message.encode('utf-8'))
        return False
    else:
        if angle_difference > 0:
            message = "LEFT"
        else:
            message = "RIGHT"
        s.send(message.encode('utf-8'))
        return True

def move_robot(distance):
    if distance > 50:
        message = "FORWARD"
        moving = True
    else:
        message = "STOP"
        moving = False
    s.send(message.encode('utf-8'))
    return moving


def main():
    video = cv2.VideoCapture(INPUT_SOURCE, cv2.CAP_DSHOW)

    model = YOLO("res/best.pt")
    robot_center = None
    arrow_center = None
    angle_deg = None
    color = (0, 0, 255)
    # Variables to store the closest ball and its distance to the robot
    closest_ball = None
    closest_ball_distance = float('inf')
    ret, frame = video.read()
    is_moving = False
    on_target = False
    message = "SPIN 40"
    s.send(message.encode('utf-8'))

    while video.isOpened():
        closest_ball = None
        closest_ball_distance = float('inf')
        ret, frame = video.read()
        if ret:

            result = model(frame, conf=CONF, iou=IOU)[0]
            detections = sv.Detections.from_yolov8(result)
            # Parse result image and calculate angle
            for i in range(len(detections)):
                xyxy = detections.xyxy[i]
                confidence = detections.confidence[i]
                class_id = detections.class_id[i]

                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2

                start_point = (int(xyxy[0]), int(xyxy[1]))
                end_point = (int(xyxy[2]), int(xyxy[3]))
                # draw/place bounding boxes on image
                # cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=2)

                if np.isin(7, class_id):
                    robot_center = (center_x, center_y)
                    color = (0, 255, 0)  # Green for robot

                elif np.isin(6, class_id):
                    arrow_center = (center_x, center_y)
                    color = (0, 0, 255)  # Red for front

                elif np.isin(1 and 2, class_id):
                    # Calculate the Euclidean distance between the robot and the ball
                    if robot_center is not None:
                        distance = math.sqrt((center_x - robot_center[0]) ** 2 + (center_y - robot_center[1]) ** 2)
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

            annotated_frame = result.plot()

            cv2.imshow("yolov8", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if closest_ball is not None and robot_center is not None and angle_deg is not None:
                is_moving = turn_robot(angle_deg, robot_center, closest_ball, is_moving)
                if is_moving == False:
                    is_moving=move_robot(closest_ball_distance)
                elif closest_ball_distance < 50:
                    message = "STOP"
                    s.send(message.encode('utf-8'))
            else:
                message = "STOP"
                s.send(message.encode('utf-8'))

    # Release the video capture object and close the display window
    video.release()
    cv2.destroyAllWindows()
    s.close()


if __name__ == "__main__":
    main()
