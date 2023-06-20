#!/usr/bin/env python3
from ev3dev2.motor import LargeMotor, OUTPUT_D, OUTPUT_A, SpeedPercent, MoveTank
from ev3dev2.motor import MediumMotor, MoveSteering, OUTPUT_B, OUTPUT_C
from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import TouchSensor, InfraredSensor
from ev3dev2.led import Leds
from ev3dev2.sound import Sound
from time import sleep
import socket
import select

# cannot be used for this version
# from pybricks.hubs import EV3Brick
# from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
#                                 InfraredSensor, UltrasonicSensor, GyroSensor)
# from pybricks.parameters import Port, Stop, Direction, Button, Color
# from pybricks.tools import wait, StopWatch, DataLog
# from pybricks.robotics import DriveBase
# from pybricks.media.ev3dev import SoundFile, ImageFile


# Create your objects here.

# This is for the 2 big motors to be able to move the robot
tank_pair = MoveTank(OUTPUT_B, OUTPUT_A)
# This is for the medium motor that control the spinner that collect/deposits the balls
scoop = MediumMotor(OUTPUT_D)


# !!!!
# link for help to
# https://github.com/ev3dev/ev3dev-lang-python
# https://ev3dev-lang.readthedocs.io/projects/python-ev3dev/en/stable/motors.html
# !!!!


# function to move the robot corresponding with the input it receives from the pc
# it gets 1 value that decides what action the robot is going to do
def move(tester):
    input = tester.split(" ")
    if (len(input) == 1):
        input.append('100')

    if (input[0] == "FORWARD"):
        tank_pair.on(SpeedPercent(-50), SpeedPercent(-50))
    elif (input[0] == "FAST"):
        tank_pair.on_for_degrees(SpeedPercent(-80), SpeedPercent(-80), 270, block=True)
    elif (input[0] == "BACK"):
        tank_pair.on(SpeedPercent(50), SpeedPercent(50))
    elif (input[0] == "BACK1"):
        tank_pair.on_for_degrees(SpeedPercent(80), SpeedPercent(80), 450, block=True)
    elif (input[0] == "RIGHT"):
        tank_pair.on(SpeedPercent(-8), SpeedPercent(8))
    elif (input[0] == "LEFT"):
        tank_pair.on(SpeedPercent(8), SpeedPercent(-8))
    elif (input[0] == "SPIN"):
        scoop.on(SpeedPercent(25))
        # scoop.on_for_degrees(SpeedPercent(25),200,block=False)
    elif (input[0] == "EJECT"):
        tank_pair.stop()
        scoop.on_for_seconds(SpeedPercent(-30), 5)
        # scoop.on_for_degrees(SpeedPercent(-30),800)
        tank_pair.on_for_rotations(SpeedPercent(80), SpeedPercent(80), 1)
        scoop.on(SpeedPercent(25), block=False)
    elif (input[0] == "SPINOFF"):
        scoop.stop()
    elif (input[0] == "DEGMOVE"):
        tank_pair.on_for_degrees(SpeedPercent(-100), SpeedPercent(-100), int(input[1]), block=False)
    elif (input[0] == "DEGTURN"):
        tank_pair.on_for_degrees(SpeedPercent(100), SpeedPercent(-100), int(input[1]), block=False)
    elif (input[0] == "STOP"):
        tank_pair.stop()
    elif (input[0] == "FIX"):
        # rotates the spinner a bit in both directions to dislodge any stuck balls, before resuming normal rotation
        scoop.on_for_degrees(SpeedPercent(-40), 40)
        scoop.on_for_degrees(SpeedPercent(70), 40)
        scoop.on(SpeedPercent(25), block=False)
    else:
        print('nope')


# this function controls the server and is also the "main" loop currently. It sets up a server and the pc connects
# to that server to be able to send inputs to the robot.
def server():
    host = "192.168.43.168"  # get local machine name
    port = 1060  # Make sure it's within the > 1024 $$ <65535 range

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))

    s.listen(1)
    client_socket, adress = s.accept()
    while True:
        if scoop.is_stalled:
            move("FIX")
        data = client_socket.recv(1024).decode('utf-8')
        if scoop.is_stalled:
            move("FIX")
        if not data:
            break
        move(data)
        data = data.upper()
        client_socket.send(data.encode('utf-8'))
        print(data)
    client_socket.close()


server()