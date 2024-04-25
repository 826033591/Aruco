###################################################################################################
###################################################################################################
###################################################################################################
#Mercury-B1_BASE_V1.0  Left_ESP32-PICO-D4 download
#GPIO4
#GPIO5
#GPIO14/TXD0
#GPIO15/RXD0
###################################################################################################

import RPi.GPIO as GPIO
import esptool
import time
import os

#GPIO4 GPIO5 EN   IO0  status
#  0     0    1    1	  use
#  0     1    0    1	  use
#  1     0    1    0	  use
#  1     1    1    1	  unuse

GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(5, GPIO.OUT)



GPIO.output(5, True)	#IO0 = HIGH
GPIO.output(4, False)	#EN = LOW,chip in reset
time.sleep(0.1)
GPIO.output(5, False)	#IO0 = LOW
GPIO.output(4, True)	#EN = HIGH,chip out of reset
time.sleep(0.05)
GPIO.output(5, True)	#IO0 = HIGH

command = ['-p', '/dev/ttyTHS0','-b','1000000', 'write_flash', '0x10000', 'left_firmware.bin']
esptool.main(command)
time.sleep(1)
os.system("sudo ./left_reset.sh")