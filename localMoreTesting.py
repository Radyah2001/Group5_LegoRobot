import socket
import cv2
import json
import math
import numpy as np
from ultralytics import YOLO
import supervision as sv

with open('VideoSourceConfig.json') as f:
    config = json.load(f)

CONF = config["CONF"]
IOU = config["IOU"]
INPUT_SOURCE = config["InputSource"]
host, port = "192.168.43.168", 1060  # get local machine name and port


def calculate_coordinates(robot_coord, target_coord):
    diff_x = target_coord[0] - robot_coord[0]
    diff_y = target_coord[1] - robot_coord[1]
    angle_rad = math.atan2(-diff_y, diff_x)
    target_angle = (math.degrees(angle_rad) + 360) % 360
    return target_angle


def move_target(robot_angle, robot_coord, target_coord, isMoving, message="STOP"):
    target_angle = calculate_coordinates(robot_coord, target_coord)
    leftDif = (target_angle - robot_angle) % 360
    rightDif = (robot_angle - target_angle) % 360
    if robot_angle - 5 < target_angle < robot_angle + 5:
        isMoving = False
    elif not isMoving:
        message = "RIGHT" if rightDif <= leftDif else "LEFT"
        isMoving = True
    return isMoving, message


def main():
    video = cv2.VideoCapture(INPUT_SOURCE, cv2.CAP_DSHOW)
    model = YOLO("res/best.pt")
    closest_ball_distance, closest_goal_distance = float('inf'), float('inf')
    isMoving = False
    robot_center, arrow_center, goal = None, None, None
    while video.isOpened():
        closest_ball = None
        closest_ball_distance = float('inf')
        bounds = []
        ret, frame = video.read()
        if ret:
            result = model(frame, conf=CONF, iou=IOU)[0]
            detections = sv.Detections.from_yolov8(result)
            robot_center, arrow_center, closest_ball = handle_detections(detections, robot_center, arrow_center,
                                                                         closest_ball, closest_ball_distance, bounds)
            goal = find_goal(goal, bounds)
            angle_deg = find_robot_angle(robot_center, arrow_center)
            if goal is not None:
                cv2.circle(frame, (int(goal[0]), int(goal[1])), radius=10, color=(0, 0, 255),
                           thickness=-1)
            cv2.imshow("yolov8", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    video.release()
    cv2.destroyAllWindows()


def handle_detections(detections, robot_center, arrow_center, closest_ball, closest_ball_distance, bounds):
    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2
        class_id = detections.class_id[i]
        if class_id == 7:
            robot_center = (center_x, center_y)
        elif class_id == 6:
            arrow_center = (center_x, center_y)
        elif class_id in [1, 2]:
            if robot_center is not None:
                distance = math.sqrt((center_x - robot_center[0]) ** 2 + (center_y - robot_center[1]) ** 2)
                if distance < closest_ball_distance:
                    closest_ball = (center_x, center_y)
                    closest_ball_distance = distance
        elif class_id == 3:
            x_center = (xyxy[0] + xyxy[2]) / 2  # calculate x center of the bound
            y_center = (xyxy[1] + xyxy[3]) / 2  # calculate y center of the bound
            bounds.append((x_center, y_center))  # save the x and y coordinates of the bounds
    return robot_center, arrow_center, closest_ball


def find_goal(goal, bounds):
    for center in bounds:
        if 200 < center[1] < 400 and center[0] > 200 and goal is None:
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


if __name__ == "__main__":
    main()
