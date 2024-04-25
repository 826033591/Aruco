from pymycobot import Mercury
import numpy as np
import time

left_arm = Mercury("/dev/ttyTHS0")
right_arm = Mercury("/dev/ttyACM0")

n = 7
# left_arm.release_servo(n)
# left_arm.focus_servo(n)
print(left_arm.set_servo_calibration(n))
