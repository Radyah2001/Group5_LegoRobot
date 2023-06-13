import socket
import time

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


def turn_robot(robot_angle, back_coord, target_coord, isMoving):
    # calculate differences in x and y coordinates
    diff_x = target_coord[0] - back_coord[0]
    diff_y = target_coord[1] - back_coord[1]

    # calculate angle in radians
    angle_rad = math.atan2(-diff_y, diff_x)

    # calculate distance
    dist = back_coord[0] - target_coord[0]

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


def move_robot(distance, target_distance, is_moving):
    if distance > target_distance:
        message = "FORWARD"
        moving = True
        s.send(message.encode('utf-8'))
        return moving
    elif is_moving == True:
        message = "STOP"
        moving = False
        s.send(message.encode('utf-8'))
        return moving
    return is_moving


'''
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

    if isMoving:
        if -2 <= angle_difference <= 2:  # Close enough to target while moving
            message = "STOP"
            s.send(message.encode('utf-8'))
            return False
    else:
        if -5 <= angle_difference <= 5:  # Not enough to target while not moving
            return False

    if angle_difference > 0:
        message = "LEFT"
    else:
        message = "RIGHT"
    s.send(message.encode('utf-8'))
    return True


def move_robot(distance, moving):
    
    if distance > 100:  # Start moving if distance is greater than 10
        message = "FORWARD 80"
        moving = True
    elif distance > 50:  # Stop moving if distance is less than 5
        message = "FORWARD 40"
        moving = True
    elif distance > 20:
        message = "FORWARD 20"
        moving = True
    else:  # Continue the previous state if the distance is between 5 and 10
        message = "STOP"
        moving = False
    s.send(message.encode('utf-8'))
    return moving
'''


def handle_detections(detections, robot_center, arrow_center, back_center, closest_ball, closest_ball_distance, bounds, cross_center):
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
        elif class_id == 3:
            x_center = (xyxy[0] + xyxy[2]) / 2  # calculate x center of the bound
            y_center = (xyxy[1] + xyxy[3]) / 2  # calculate y center of the bound
            bounds.append((x_center, y_center))  # save the x and y coordinates of the bounds
    return robot_center, arrow_center, closest_ball, back_center


def calcBallDist(ball, frontArrow):
    return math.sqrt((frontArrow[0] - ball[0]) ** 2 + (frontArrow[1] - ball[1]) ** 2)


def find_goal(goal, bounds):
    for center in bounds:
        if 200 < center[1] < 400 and center[0] > 200 and goal is None:
            goal = (center[0], center[1])
    return goal


def find_robot_angle(back_center, arrow_center):
    if back_center and arrow_center:
        diff_x = arrow_center[0] - back_center[0]
        diff_y = arrow_center[1] - back_center[1]
        angle_rad = math.atan2(-diff_y, diff_x)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        print("Robot is facing at angle:", angle_deg, "degrees")
        return angle_deg
    return None


def main():
    video = cv2.VideoCapture(INPUT_SOURCE, cv2.CAP_DSHOW)
    model = YOLO("res/best.pt")
    closest_ball_saved = None
    robot_center, arrow_center, back_center, goal = None, None, None, None
    is_moving = False
    bounds = []
    s.send("SPIN".encode('utf-8'))

    while video.isOpened():
        closest_ball, closest_ball_distance = None, float('inf')
        ret, frame = video.read()

        if ret:
            result = model(frame, conf=CONF, iou=IOU)[0]
            detections = sv.Detections.from_yolov8(result)
            robot_center, arrow_center, closest_ball, back_center = handle_detections(detections, robot_center,
                                                                                      arrow_center, back_center,
                                                                                      closest_ball,
                                                                                      closest_ball_distance, bounds)
            goal = find_goal(goal, bounds)
            angle_deg = find_robot_angle(back_center, arrow_center)

            if goal:
                cv2.circle(frame, (int(goal[0]), int(goal[1])), radius=10, color=(0, 0, 255), thickness=-1)

            annotated_frame = result.plot()
            cv2.imshow("yolov8", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if all([closest_ball, back_center, angle_deg]):
                closest_ball_saved = closest_ball if closest_ball_saved is None else closest_ball_saved
                is_moving = turn_robot(angle_deg, back_center, closest_ball_saved, is_moving)
                dist_to_ball = calcBallDist(closest_ball_saved, arrow_center)

                if not is_moving:
                    is_moving = move_robot(dist_to_ball, 5, is_moving)

                if dist_to_ball <= 20:
                    s.send("FORWARD".encode('utf-8'))

                if dist_to_ball <= 10:
                    closest_ball_saved = None
                    s.send("STOP".encode('utf-8'))

            elif closest_ball is None and goal:
                is_moving = turn_robot(angle_deg, back_center, goal, is_moving)
                dist_to_goal = calcBallDist(goal, arrow_center)

                if not is_moving:
                    is_moving = move_robot(dist_to_goal, 50, is_moving)

                if dist_to_goal <= 50:
                    s.send("EJECT".encode('utf-8'))
                    closest_ball_saved = None

            else:
                s.send("STOP".encode('utf-8'))

    video.release()
    cv2.destroyAllWindows()
    s.close()

if __name__ == "__main__":
    main()
