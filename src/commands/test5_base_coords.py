from pymycobot import Mercury
from cobotx_frame_converter import *
import time
import numpy as np

left_arm = Mercury("/dev/ttyTHS0")
right_arm = Mercury("/dev/ttyACM0")

arm = left_arm
coords = arm.get_base_coords()
print(coords)
x,y,z,rx,ry,rz = coords
x+=0
y+=0
z = 350
rz = 0
coords = [x,y,z,rx,ry,rz]
print(coords)
arm.send_base_coords(coords, 50)

