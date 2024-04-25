
import esptool
import time
import os
#GPIO23 GPIO24 EN   IO0
#  0     0     1    1
#  0     1     0    1
#  1     0     1    0
#  1     1     1    1


command = ['-p', '/dev/ttyACM0','-b','1000000', 'write_flash', '0x10000', 'right_firmware.bin']
esptool.main(command)

time.sleep(1)
os.system("sudo ./right_reset.sh")