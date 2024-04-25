from pymycobot import Mercury
from arm_control import *

#left_arm = Mercury("/dev/ttyTHS0")
right_arm = Mercury("/dev/ttyACM0")

# ttyACM1
# right_arm

right_arm.send_angles([0.0, 45.0, 0.0, -90.0, 90.0, 90.0, 0.0], 30)