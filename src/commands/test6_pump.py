from pymycobot import Mercury
from arm_control import pump_on, pump_off
import time

arm = Mercury("/dev/ttyTHS0")

# print("start")
# pump_on(arm)
# time.sleep(10)
# # print("end")

# pump_off(arm)
# time.sleep(3)

arm.set_gripper_value(100, 50)
# arm.set_gripper_state()