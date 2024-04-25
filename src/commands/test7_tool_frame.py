from pymycobot import Mercury
from arm_control import *

left_arm = Mercury("/dev/ttyTHS0")
right_arm = Mercury("/dev/ttyACM0")

left_arm.send_angles([0.01, 35.0, 0.0, -134.99, 89.99, 90.0, -14.99], 20)
time.sleep(5)


left_arm.set_tool_reference([0, 0, 64, 0, 0, 0])
time.sleep(0.03)
left_arm.set_movement_type(1)
time.sleep(0.03)
left_arm.set_end_type(1)
time.sleep(0.03)

left_arm.send_coords([214.1, 55.0, 92.0, -90.0, 65.0, 0.02], 10)
time.sleep(5)
