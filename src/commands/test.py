from pymycobot import Mercury
import time

left_arm = Mercury("/dev/ttyTHS0", debug=True)
right_arm = Mercury("/dev/ttyACM1", debug=True)

print(left_arm.get_system_version())
print(right_arm.get_robot_version())