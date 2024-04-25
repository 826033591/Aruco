from pymycobot import Mercury
from arm_control import *

left_arm = Mercury("/dev/ttyTHS0")
right_arm = Mercury("/dev/ttyACM0")

#left_arm.send_base_coords([378.8, -10.8, 375.4, 180, 0.0, 0.06], 80)
# left_arm.send_angles([0,0,0,0,0,0,0], 80)
# left_arm.send_angles([-29.96, 57.82, 0.0, -63.23, 73.5, 64.58, -64.84], 50)
# left_arm.send_angles([0,0,0,0,0,90,0], 80)
# left_arm.release_all_servos()
# right_arm.release_servo(10)
# left_arm.release_servo(7)
# left_arm.set_servo_calibration(1)
# left_arm.set_servo_calibration(2)
# left_arm.set_servo_calibration(3)
# left_arm.set_servo_calibration(4)
# left_arm.set_servo_calibration(5)
# left_arm.set_servo_calibration(6)
# left_arm.set_servo_calibration(7)

# left_arm.focus_servo(1)
# left_arm.focus_servo(2)
# left_arm.focus_servo(3)
# left_arm.focus_servo(4)
# left_arm.focus_servo(5)
# left_arm.focus_servo(6)
# left_arm.focus_servo(7)

# left_arm.power_on()
# print(left_arm.get_robot_status())
# print(left_arm.servo_restore(6))
