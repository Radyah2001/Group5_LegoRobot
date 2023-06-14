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

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))

'''
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
    if 5 < distance < 20:
        message = "FAST"
        moving = True
        s.send(message.encode('utf-8'))
        return moving
    elif distance > target_distance:
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


def navigate_robot(robot_angle, back_coord, target_coord, distance, target_distance, is_moving, obstacle_avoidance,arrow_center, obstacle_coord=None):
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
        is_moving = False
    else:
        if angle_difference > 0:
            message = "LEFT"
        else:
            message = "RIGHT"
        s.send(message.encode('utf-8'))
        is_moving = True

    # Obstacle avoidance logic
    if obstacle_coord is not None:
        obstacle_dist = math.sqrt((arrow_center[0] - obstacle_coord[0]) ** 2 + (arrow_center[1] - obstacle_coord[1]) ** 2)
        if obstacle_dist < 30:  # threshold distance to avoid obstacle
            if not obstacle_avoidance:
                if angle_difference > 0:
                    message = "RIGHT"  # turn right to avoid obstacle
                else:
                    message = "LEFT"  # turn left to avoid obstacle
                s.send(message.encode('utf-8'))
                is_moving = True
                obstacle_avoidance = True
            else:  # If we have already turned, let's move forward
                message = "FORWARD"
                s.send(message.encode('utf-8'))
                is_moving = True
                obstacle_avoidance = False

    # Only move if robot is already facing target
    if not is_moving and not obstacle_avoidance:
        if 5 < distance < 15:
            message = "FAST"
            is_moving = True
            s.send(message.encode('utf-8'))
        elif distance > target_distance:
            message = "FORWARD"
            is_moving = True
            s.send(message.encode('utf-8'))
        else:
            message = "STOP"
            is_moving = False
            s.send(message.encode('utf-8'))

    return is_moving, obstacle_avoidance


def handle_detections(detections, robot_center, arrow_center, back_center, closest_ball, closest_ball_distance, bounds,
                      cross_center):
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
        elif class_id == 5:
            cross_center = (center_x, center_y)
    return robot_center, arrow_center, closest_ball, back_center, cross_center


def calcDist(target, frontArrow):
    return math.sqrt((frontArrow[0] - target[0]) ** 2 + (frontArrow[1] - target[1]) ** 2)


def find_goal(goal, bounds, checkpoint):
    for center in bounds:
        if 200 < center[1] < 400 and center[0] > 200 and goal is None:
            goal = (center[0], center[1])
            checkpoint = (goal[0] - 100, goal[1])
    return goal, checkpoint


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
    closest_ball_distance, closest_goal_distance = float('inf'), float('inf')
    checkpoint_reached = False
    closest_ball_saved = None
    is_moving = False
    robot_center, arrow_center, back_center, goal, cross_center, checkpoint = None, None, None, None, None, None
    message = "SPIN"
    s.send(message.encode('utf-8'))
    obstacle_avoidance = False
    while video.isOpened():
        closest_ball = None
        closest_ball_distance = float('inf')
        bounds = []
        ret, frame = video.read()
        if ret:
            result = model(frame, conf=CONF, iou=IOU)[0]
            detections = sv.Detections.from_yolov8(result)
            robot_center, arrow_center, closest_ball, back_center, cross_center = handle_detections(detections,
                                                                                                    robot_center,
                                                                                                    arrow_center,
                                                                                                    back_center,
                                                                                                    closest_ball,
                                                                                                    closest_ball_distance,
                                                                                                    bounds,
                                                                                                    cross_center)
            goal, checkpoint = find_goal(goal, bounds, checkpoint)
            angle_deg = find_robot_angle(back_center, arrow_center)
            if goal is not None:
                cv2.circle(frame, (int(goal[0]), int(goal[1])), radius=10, color=(0, 0, 255),
                           thickness=-1)
                cv2.circle(frame, (int(checkpoint[0]), int(checkpoint[1])), radius=15, color=(0, 250, 250),
                           thickness=-1)
            annotated_frame = result.plot()
            cv2.imshow("yolov8", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if closest_ball is not None and back_center is not None and angle_deg is not None:
                if closest_ball_saved is None:
                    closest_ball_saved = closest_ball
                if calcDist(closest_ball_saved, arrow_center) > calcDist(closest_ball_saved, robot_center) and calcDist(
                        closest_ball_saved, robot_center) <= 50:
                    message = "BACK"
                    s.send(message.encode('utf-8'))
                is_moving, obstacle_avoidance = navigate_robot(angle_deg, back_center, closest_ball_saved,
                                           calcDist(closest_ball_saved, arrow_center), 5, is_moving, obstacle_avoidance,arrow_center, cross_center)
                # elif calcBallDist(closest_ball_saved, arrow_center) <= 20:
                #    message = "FORWARD"
                #    s.send(message.encode('utf-8'))
                if calcDist(closest_ball_saved, arrow_center) <= 5:
                    closest_ball_saved = None
                    message = "STOP"
                    s.send(message.encode('utf-8'))
            elif closest_ball is None and goal is not None:
                if not checkpoint_reached:
                    is_moving, obstacle_avoidance = navigate_robot(angle_deg, back_center, checkpoint, calcDist(checkpoint, arrow_center),
                                               15, is_moving, obstacle_avoidance,arrow_center, cross_center)
                    if calcDist(checkpoint, arrow_center) <= 15:
                        checkpoint_reached = True
                else:
                    is_moving, obstacle_avoidance = navigate_robot(angle_deg, back_center, goal, calcDist(goal, arrow_center), 30, is_moving,obstacle_avoidance,arrow_center, cross_center)
                    if calcDist(goal, arrow_center) <= 30:
                        message = "EJECT"
                        s.send(message.encode('utf-8'))
                        closest_ball_saved = None
                        print("distance to goal is:", calcDist(goal, arrow_center))
            else:
                message = "STOP"
                s.send(message.encode('utf-8'))
    video.release()
    cv2.destroyAllWindows()
    s.close()


if __name__ == "__main__":
    main()
