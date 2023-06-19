import socket
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import math
import json

# Load the config for the video source
with open('VideoSourceConfig.json') as f:
    config = json.load(f)
CONF = config["CONF"]
IOU = config["IOU"]
INPUT_SOURCE = config["InputSource"]

# Connecting to robot server
host = "192.168.43.168"  # get local machine name
port = 1060  # Make sure it's within the > 1024 $$ <65535 range
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))

# Function to turn the robot based on angle difference
def turn_robot(angle_difference):
    turning = False
    if angle_difference > 2:
        message = "LEFT"
        turning = True
    elif angle_difference < 2:
        message = "RIGHT"
        turning = True
    else:
        message = "STOP"
        turning = False
    s.send(message.encode('utf-8'))
    return turning

# Funktion to move the robot based on the distance to target
def move_robot(distance, target_distance, is_moving):
    if 5 < distance < 25:
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

# Checking the angle between the robot and the target
def checkAngle(robot_angle, target_coord, back_coord):
    # calculate differences in x and y coordinates
    diff_x = target_coord[0] - back_coord[0]
    diff_y = target_coord[1] - back_coord[1]
    # calculate angle in radians
    angle_rad = math.atan2(-diff_y, diff_x)
    # convert angle to degrees
    target_angle = (math.degrees(angle_rad) + 360) % 360

    angle_difference = target_angle - robot_angle
    if angle_difference > 180:
        angle_difference -= 360
    elif angle_difference < -180:
        angle_difference += 360

    if -2 <= angle_difference <= 2:  # Close enough to target
        return True, angle_difference
    else:
        return False, angle_difference


# This funktion processes the detected objects from the YOLO model, and returns detections as objects
def handle_detections(detections, robot_center, arrow_center, back_center, bounds,
                      cross_center, balls):
    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2
        class_id = detections.class_id[i]
        if class_id == 7: #robot center
            robot_center = (center_x, center_y)
        elif class_id == 6: #arrow center
            arrow_center = (center_x, center_y)
        elif np.isin(0, class_id): #back center
            back_center = (center_x, center_y)
        elif class_id in [1, 2]: #balls
            balls.append((center_x, center_y))
        elif class_id == 3: #bounds
            x_center = (xyxy[0] + xyxy[2]) / 2  # calculate x center of the bound
            y_center = (xyxy[1] + xyxy[3]) / 2  # calculate y center of the bound
            bounds.append((x_center, y_center))  # save the x and y coordinates of the bounds
        elif class_id == 5: #cross
            cross_center = (center_x, center_y)
    return robot_center, arrow_center, back_center, cross_center


# This funktion finds the closest ball, and exclude some that are to close to bounds, or cross.
def calc_closest_ball(balls, north, west, south, east, robot_center, closest_ball, closest_ball_distance, cross_center):
    for ball in balls:
        distance = math.sqrt((ball[0] - robot_center[0]) ** 2 + (ball[1] - robot_center[1]) ** 2)
        if distance < closest_ball_distance and west[0] + 30 < ball[0] < east[0] - 30 and south[1] - 30 > ball[1] > \
                north[1] + 30 and calcDist(cross_center, ball) > 75:
            closest_ball = (ball[0], ball[1])
            closest_ball_distance = distance
    return closest_ball

# This funktion is used if we can't find bounds, so we just move to the closest ball with no restrictions
def calc_closest_ball_without_directions(balls, closest_ball, closest_ball_distance, robot_center):
    for ball in balls:
        distance = math.sqrt((ball[0] - robot_center[0]) ** 2 + (ball[1] - robot_center[1]) ** 2)
        if distance < closest_ball_distance:
            closest_ball = (ball[0], ball[1])
            closest_ball_distance = distance
    return closest_ball

# Calculates the Euclidean distance between two objects
def calcDist(target, frontArrow):
    return math.sqrt((frontArrow[0] - target[0]) ** 2 + (frontArrow[1] - target[1]) ** 2)

# Finds the goal to the right of the screen
def find_goal(goal, bounds, checkpoint):
    for center in bounds:
        if 200 < center[1] < 400 and center[0] > 200 and goal is None:
            goal = (center[0], center[1])
            checkpoint = (goal[0] - 100, goal[1])
    return goal, checkpoint

# Calculates the robots angle that it is currently facing
def find_robot_angle(back_center, arrow_center):
    if back_center and arrow_center:
        diff_x = arrow_center[0] - back_center[0]
        diff_y = arrow_center[1] - back_center[1]
        angle_rad = math.atan2(-diff_y, diff_x)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        print("Robot is facing at angle:", angle_deg, "degrees")
        return angle_deg
    return None

# Navigates the robot based on angle and distance to target
def navigate_robot(robot_angle, back_coord, target_coord, distance, target_distance, is_moving):
    if back_coord is not None:
        onTarget, angleDif = checkAngle(robot_angle, target_coord, back_coord)
        if onTarget:
            if is_moving == False:
                message = "STOP"
                s.send(message.encode('utf-8'))
            is_moving = move_robot(distance, target_distance, is_moving)
        else:
            is_turning = turn_robot(angleDif)
            is_moving = False
    return is_moving

# Calculates the second-closest offset from the cross to navigate around it
def get_multiple_closest_offsets(cross_center, target, offset):
    offsets = [
        (cross_center[0], cross_center[1] + offset),  # North
        (cross_center[0], cross_center[1] - offset),  # South
        (cross_center[0] + offset, cross_center[1]),  # East
        (cross_center[0] - offset, cross_center[1]),  # West
        (cross_center[0] + offset, cross_center[1] + offset),  # North-East
        (cross_center[0] + offset, cross_center[1] - offset),  # South-East
        (cross_center[0] - offset, cross_center[1] + offset),  # North-West
        (cross_center[0] - offset, cross_center[1] - offset)  # South-West
    ]
    distances = [calcDist(target, offset) for offset in offsets]
    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
    second_closest_offset = offsets[sorted_indices[1]]
    fourth_closest_offset = offsets[sorted_indices[3]]
    sixth_closest_offset = offsets[sorted_indices[5]]

    return second_closest_offset, fourth_closest_offset, sixth_closest_offset

# Finds which bound is where, so we know which direction to offset
def get_north_east_south_west(bounds, east, west, north, south):
    for center in bounds:
        if 200 < center[1] < 400 and center[0] > 200:
            east = (center[0], center[1])
        elif 200 < center[1] < 400 and center[0] < 200:
            west = (center[0], center[1])
        elif 200 < center[0] < 400 and center[1] > 200:
            south = (center[0], center[1])
        elif 200 < center[0] < 400 and center[1] < 200:
            north = (center[0], center[1])
    return east, west, north, south


# Main method that runs the core logic
def main():
    video = cv2.VideoCapture(INPUT_SOURCE, cv2.CAP_DSHOW)
    model = YOLO("res/best.pt")
    closest_ball_distance, closest_goal_distance = float('inf'), float('inf')
    checkpoint_reached = False
    closest_ball_saved = None
    is_moving = False
    robot_center, arrow_center, back_center, goal, cross_center, checkpoint = None, None, None, None, None, None
    north, east, south, west = None, None, None, None
    message = "SPIN"
    s.send(message.encode('utf-8'))
    offset3, offset2, offset1 = None, None, None
    goToOffset = False
    north, east, south, west = None, None, None, None

    #Main loop
    while video.isOpened():
        closest_ball = None
        closest_ball_distance = float('inf')
        bounds = []
        balls = []
        # Read frame from video
        ret, frame = video.read()
        if ret:
            # Run the YOLO model ont the frame
            result = model(frame, conf=CONF, iou=IOU)[0]
            detections = sv.Detections.from_yolov8(result)

            # Saves the objects that was detected by the YOLO model
            robot_center, arrow_center, back_center, cross_center = handle_detections(detections, robot_center, arrow_center,
                                                                                      back_center, bounds, cross_center, balls)
            east, west, north, south = get_north_east_south_west(bounds, east, west, north, south)

            # Finds the closest ball
            if north is not None and west is not None and east is not None and south is not None and robot_center is not None:
                closest_ball = calc_closest_ball(balls, north, west, south, east, robot_center, closest_ball,
                                                 closest_ball_distance, cross_center)
            elif robot_center is not None:
                calc_closest_ball_without_directions(balls, closest_ball, closest_ball_distance, robot_center)

            goal, checkpoint = find_goal(goal, bounds, checkpoint)
            angle_deg = find_robot_angle(back_center, arrow_center)
            if east is not None and west is not None and north is not None and south is not None:
                if west[0] + 20 > arrow_center[0] or arrow_center[0] > east[0] - 20 or south[1] - 20 < arrow_center[1] \
                        or arrow_center[1] < north[1] + 20:
                    message = "BACK1"
                    s.send(message.encode('utf-8'))
                if west[0] + 20 > back_center[0] or back_center[0] > east[0] - 20 or south[1] - 20 < back_center[1] \
                        or back_center[1] < north[1] + 20:
                    message = "FAST"
                    s.send(message.encode('utf-8'))
                cv2.circle(frame, (int(east[0]), int(east[1])), radius=10, color=(0, 0, 255), thickness=-1)
                cv2.circle(frame, (int(west[0]), int(west[1])), radius=10, color=(0, 0, 255), thickness=-1)
                cv2.circle(frame, (int(north[0]), int(north[1])), radius=10, color=(0, 0, 255), thickness=-1)
                cv2.circle(frame, (int(south[0]), int(south[1])), radius=10, color=(0, 0, 255), thickness=-1)
            '''if cross_center is not None and closest_ball is not None:
                offset = get_second_closest_offset(cross_center, closest_ball, 135)
                cv2.circle(frame, (int(offset[0]), int(offset[1])), radius=10, color=(0, 0, 255),
                           thickness=-1)'''
            if goal is not None:
                cv2.circle(frame, (int(goal[0]), int(goal[1])), radius=10, color=(0, 0, 255),
                           thickness=-1)
                cv2.circle(frame, (int(checkpoint[0]), int(checkpoint[1])), radius=15, color=(0, 250, 250),
                           thickness=-1)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if closest_ball_saved is not None:
                if calcDist(cross_center, arrow_center) <= 75 and calcDist(closest_ball_saved, arrow_center) > calcDist(
                        closest_ball_saved, cross_center):  # When front of robot is close to cross_center
                    offset3, offset2, offset1 = get_multiple_closest_offsets(cross_center, closest_ball_saved, 100)
                    goToOffset1 = True
                    message = "BACK1"
                    s.send(message.encode('utf-8'))
                if offset1 is not None and goToOffset1:
                    is_moving = navigate_robot(angle_deg, back_center, offset1,
                                               calcDist(offset1, arrow_center), 10, is_moving)
                    if calcDist(offset1, arrow_center) < 10:
                        goToOffset1 = False
                        goToOffset2 = True
                    continue
                if offset2 is not None and goToOffset2:
                    is_moving = navigate_robot(angle_deg, back_center, offset2,
                                               calcDist(offset2, arrow_center), 10, is_moving)
                    if calcDist(offset2, arrow_center) < 10:
                        goToOffset2 = False
                        goToOffset3 = True
                    continue
                if offset3 is not None and goToOffset3:
                    is_moving = navigate_robot(angle_deg, back_center, offset3,
                                               calcDist(offset3, arrow_center), 10, is_moving)
                    if calcDist(offset3, arrow_center) < 10:
                        goToOffset3 = False
                    continue


                if calcDist(closest_ball_saved, arrow_center) > calcDist(closest_ball_saved, robot_center) and calcDist(
                        closest_ball_saved, robot_center) <= 50:
                    message = "BACK1"
                    s.send(message.encode('utf-8'))
                is_moving = navigate_robot(angle_deg, back_center, closest_ball_saved,
                                           calcDist(closest_ball_saved, arrow_center), 5, is_moving)
                cv2.circle(frame, (int(closest_ball_saved[0]), int(closest_ball_saved[1])), radius=10, color=(0, 0, 255),
                           thickness=-1)
                if calcDist(closest_ball_saved, arrow_center) <= 10:
                    closest_ball_saved = None
                    message = "STOP"
                    s.send(message.encode('utf-8'))
                continue
            if closest_ball is not None and back_center is not None and angle_deg is not None:
                if closest_ball_saved is None:
                    closest_ball_saved = closest_ball
            elif closest_ball_saved is None and goal is not None:
                if not checkpoint_reached:
                    if calcDist(cross_center, arrow_center) <= 75 and calcDist(checkpoint, arrow_center) > calcDist(checkpoint, cross_center):  # When front of robot is close to cross_center
                        offset3, offset2, offset1 = get_multiple_closest_offsets(cross_center, checkpoint, 100)
                        goToOffset1 = True
                        message = "BACK1"
                        s.send(message.encode('utf-8'))
                    if offset1 is not None and goToOffset1:
                        is_moving = navigate_robot(angle_deg, back_center, offset1,
                                                   calcDist(offset1, arrow_center), 10, is_moving)
                        if calcDist(offset1, arrow_center) < 10:
                            goToOffset1 = False
                            goToOffset2 = True
                        continue
                    if offset2 is not None and goToOffset2:
                        is_moving = navigate_robot(angle_deg, back_center, offset2,
                                                   calcDist(offset2, arrow_center), 10, is_moving)
                        if calcDist(offset2, arrow_center) < 10:
                            goToOffset2 = False
                            goToOffset3 = True
                        continue
                    if offset3 is not None and goToOffset3:
                        is_moving = navigate_robot(angle_deg, back_center, offset3,
                                                   calcDist(offset3, arrow_center), 10, is_moving)
                        if calcDist(offset3, arrow_center) < 10:
                            goToOffset3 = False
                        continue
                    is_moving = navigate_robot(angle_deg, back_center, checkpoint, calcDist(checkpoint, arrow_center),
                                               15, is_moving)
                    if calcDist(checkpoint, arrow_center) <= 15:
                        checkpoint_reached = True
                else:
                    is_moving = navigate_robot(angle_deg, back_center, goal, calcDist(goal, arrow_center), 30,
                                               is_moving)
                    if calcDist(goal, arrow_center) <= 30:
                        message = "EJECT"
                        s.send(message.encode('utf-8'))
                        closest_ball_saved = None
                        print("distance to goal is:", calcDist(goal, arrow_center))
            else:
                message = "STOP"
                s.send(message.encode('utf-8'))

            annotated_frame = result.plot()
            cv2.imshow("yolov8", annotated_frame)
    video.release()
    cv2.destroyAllWindows()
    s.close()


if __name__ == "__main__":
    main()
