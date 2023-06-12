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

    if -3 <= angle_difference <= 3:  # Close enough to target
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
    if distance > 5:
        message = "FORWARD"
        moving = True
    else:
        message = "STOP"
        moving = False
    s.send(message.encode('utf-8'))
    return moving



def handle_detections(detections, robot_center, arrow_center, back_center, closest_ball, closest_ball_distance):
    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2
        class_id = detections.class_id[i]
        if class_id == 7:
            robot_center = (center_x, center_y)
        elif class_id == 6:
            arrow_center = (center_x, center_y)
        elif np.isin(0, class_id):
            back_center = (center_x, center_y)
        elif class_id in [1, 2]:
            if robot_center is not None:
                distance = math.sqrt((center_x - robot_center[0]) ** 2 + (center_y - robot_center[1]) ** 2)
                if distance < closest_ball_distance:
                    closest_ball = (center_x, center_y)
                    closest_ball_distance = distance
    return robot_center, arrow_center, closest_ball

def calcBallDist(ball, frontArrow):
    return math.sqrt((frontArrow[0] - ball[0]) ** 2 + (frontArrow[1] - ball[1]) ** 2)


def find_goal(goal, bounds):
    for center in bounds:
        if 200 < center[1] < 400 and center[0] < 200 and goal is None:
            goal = (center[0], center[1])
    return goal

def find_robot_angle(robot_center, arrow_center):
    if robot_center and arrow_center:
        diff_x = arrow_center[0] - robot_center[0]
        diff_y = arrow_center[1] - robot_center[1]
        angle_rad = math.atan2(-diff_y, diff_x)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        print("Robot is facing at angle:", angle_deg, "degrees")
        return angle_deg
    return None

def main():
    video = cv2.VideoCapture(INPUT_SOURCE, cv2.CAP_DSHOW)
    model = YOLO("res/best.pt")
    closest_ball_distance, closest_goal_distance = float('inf'), float('inf')
    closest_ball_saved = None
    is_moving = False
    robot_center, arrow_center, back_center, goal = None, None, None, None
    message = "SPIN 40"
    s.send(message.encode('utf-8'))
    while video.isOpened():
        closest_ball = None
        closest_ball_distance = float('inf')
        bounds = []
        ret, frame = video.read()
        if ret:
            result = model(frame, conf=CONF, iou=IOU)[0]
            detections = sv.Detections.from_yolov8(result)
            robot_center, arrow_center, closest_ball = handle_detections(detections, robot_center, arrow_center,
                                                                     back_center, closest_ball, closest_ball_distance)
            goal = find_goal(goal, bounds)
            angle_deg = find_robot_angle(robot_center, arrow_center)
            if goal is not None:
                cv2.circle(frame, (int(goal[0]), int(goal[1])), radius=10, color=(0, 0, 255),
                           thickness=-1)
            annotated_frame = result.plot()
            cv2.imshow("yolov8", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if closest_ball is not None and robot_center is not None and angle_deg is not None:
                if closest_ball_saved is None:
                    closest_ball_saved = closest_ball
                is_moving = turn_robot(angle_deg, robot_center, closest_ball_saved, is_moving)
                if is_moving == False:
                    is_moving = move_robot(calcBallDist(closest_ball_saved, arrow_center))
                elif calcBallDist(closest_ball_saved, arrow_center) <= 20:
                    message = "FORWARD"
                    s.send(message.encode('utf-8'))
                if calcBallDist(closest_ball_saved, arrow_center) <= 10:
                    closest_ball_saved = None
                    message = "STOP"
                    s.send(message.encode('utf-8'))
            else:
                message = "STOP"
                s.send(message.encode('utf-8'))
    video.release()
    cv2.destroyAllWindows()
    s.close()


if __name__ == "__main__":
    main()
