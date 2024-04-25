from pymycobot import Mercury
from arm_control import *

left_arm = Mercury("/dev/ttyTHS0")
#right_arm = Mercury("/dev/ttyACM1")

coords = left_arm.get_base_coords()
x,y,z,rx,ry,rz = coords

z += -50
rx = 180
ry = 0
rz = 0

left_arm.send_base_coords([x,y,z,rx,ry,rz], 50)