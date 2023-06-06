#!/usr/bin/env python3
from ev3dev2.motor import LargeMotor, OUTPUT_D, OUTPUT_A, SpeedPercent, MoveTank
from ev3dev2.motor import MediumMotor, MoveSteering,OUTPUT_B, OUTPUT_C
from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import TouchSensor, InfraredSensor
from ev3dev2.led import Leds
from ev3dev2.sound import Sound
from time import sleep
import socket



#cannot be used for this version
# from pybricks.hubs import EV3Brick
# from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
#                                 InfraredSensor, UltrasonicSensor, GyroSensor)
# from pybricks.parameters import Port, Stop, Direction, Button, Color
# from pybricks.tools import wait, StopWatch, DataLog
# from pybricks.robotics import DriveBase
# from pybricks.media.ev3dev import SoundFile, ImageFile




# Create your objects here.

#This is for the 2 big motors to be able to move the robot
tank_pair = MoveTank(OUTPUT_B,OUTPUT_A)
scoop = MediumMotor(OUTPUT_D)



#sound = Sound()
#opts = '-a 200 -s 130 -v'




# Write your program here.
#sound.tone([
#    (392, 350, 100), (392, 350, 100), (392, 350, 100), (311.1, 250, 100),
#    (466.2, 25, 100), (392, 350, 100), (311.1, 250, 100), (466.2, 25, 100),
#    (392, 700, 100), (587.32, 350, 100), (587.32, 350, 100),
#    (587.32, 350, 100), (622.26, 250, 100), (466.2, 25, 100),
#    (369.99, 350, 100), (311.1, 250, 100), (466.2, 25, 100), (392, 700, 100),
#    (784, 350, 100), (392, 250, 100), (392, 25, 100), (784, 350, 100),
#    (739.98, 250, 100), (698.46, 25, 100), (659.26, 25, 100),
#    (622.26, 25, 100), (659.26, 50, 400), (415.3, 25, 200), (554.36, 350, 100),
#    (523.25, 250, 100), (493.88, 25, 100), (466.16, 25, 100), (440, 25, 100),
#    (466.16, 50, 400), (311.13, 25, 200), (369.99, 350, 100),
#    (311.13, 250, 100), (392, 25, 100), (466.16, 350, 100), (392, 250, 100),
#    (466.16, 25, 100), (587.32, 700, 100), (784, 350, 100), (392, 250, 100),
#    (392, 25, 100), (784, 350, 100), (739.98, 250, 100), (698.46, 25, 100),
#    (659.26, 25, 100), (622.26, 25, 100), (659.26, 50, 400), (415.3, 25, 200),
#    (554.36, 350, 100), (523.25, 250, 100), (493.88, 25, 100),
#    (466.16, 25, 100), (440, 25, 100), (466.16, 50, 400), (311.13, 25, 200),
#    (392, 350, 100), (311.13, 250, 100), (466.16, 25, 100),
#    (392.00, 300, 150), (311.13, 250, 100), (466.16, 25, 100), (392, 700)
#    ],play_type=0)
run = 1
#sound.speak('klokken er 12 37, og du er fanget', espeak_opts=opts+'da')
#tank_pair.on(left_speed=0,right_speed=0)



#!!!!
#link for help to
#https://github.com/ev3dev/ev3dev-lang-python
#https://ev3dev-lang.readthedocs.io/projects/python-ev3dev/en/stable/motors.html
#!!!!


#function to move the robot corresponding with the input it receives from the pc
#it gets 2 values, the first decides what action to take, and the second value decides for how long to run that action
def move(tester):
  input = tester.split(" ")
  if(len(input)==1):
    input.append('100')
  if(input[0]=="FORWARD"):
      tank_pair.on_for_rotations(SpeedPercent(-100),SpeedPercent(-100),int(input[1]),block=False)
  elif(input[0]=="BACK"):
      tank_pair.on_for_rotations(SpeedPercent(100),SpeedPercent(100),int(input[1]),block=False)
  elif(input[0] == "RIGHT"):
      tank_pair.on_for_rotations(SpeedPercent(-50),SpeedPercent(50),int(input[1]),block=False)
  elif(input[0] == "LEFT"):
      tank_pair.on_for_rotations(SpeedPercent(50),SpeedPercent(-50),int(input[1]),block=False)
  elif(input[0] == "SCOOP"):
      scoop.on(SpeedPercent(int(input[1])))
  elif(input[0] == "SCOOPOFF"):
      scoop.stop()
  elif(input[0] == "DEGMOVE"):
      tank_pair.on_for_degrees(SpeedPercent(-100),SpeedPercent(-100),int(input[1]),block=False)
  elif(input[0] == "DEGTURN"):
      tank_pair.on_for_degrees(SpeedPercent(100),SpeedPercent(-100),int(input[1]),block=False)
  elif(input[0] == "STOP"):
      tank_pair.stop()
  elif(input[0] == "TEST"):
      tank_pair.on(SpeedPercent(100),SpeedPercent(-100))
  else:
      print('nope')


#this function controls the server and is also the "main" loop currently. It sets up a server and the pc connects
#to that server to be able to send inputs to the robot.
def server():
  host = "192.168.43.168"   # get local machine name
  port = 1060  # Make sure it's within the > 1024 $$ <65535 range
  
  s = socket.socket()
  s.bind((host, port))
  
  s.listen(1)
  client_socket, adress = s.accept()
  while True:
    data = client_socket.recv(1024).decode('utf-8')
    if not data:
      break
    data = data.upper()
    client_socket.send(data.encode('utf-8'))
    move(data)
    print(data)
  client_socket.close()


server()



#while (run == 1):
#    tankpair.on_for_seconds(SpeedPercent(-100),SpeedPercent(-100),3)