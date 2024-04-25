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
# os.system("sudo ./left_flash.sh")

#GPIO.output(5, True)	#IO0 = HIGH
#GPIO.output(4, False)	#EN = LOW,chip in reset
#time.sleep(0.1)
#GPIO.output(4, True)	#EN = HIGH,chip out of reset

###################################################################################################
###################################################################################################
###################################################################################################







#G0 = 4
#EN = 5

#DTR = 5
#RTS = 4
#_false = 1
#_true = 0


#GPIO.output(DTR, _false)	#IO0 = HIGH
#GPIO.output(RTS, _true)	#EN = LOW,chip in reset
#time.sleep(0.1)
#GPIO.output(DTR, _true)
#GPIO.output(RTS, _false)
#time.sleep(0.05)
#GPIO.output(DTR, _false)

#command = ['-p', '/dev/ttyTHS0','-b','1000000', 'write_flash', '0x10000', 'MercuryB1R.bin']
#esptool.main(command)

#two lines below Means that EN is 0 and IO0 is 1
#GPIO.output(RTS, _false)  #4--1  RTS--0
#GPIO.output(DTR, _true)   #5--0  DTR--1

#time.sleep(0.1)

#GPIO.output(DTR, _false)
#GPIO.output(G0, 1)
#GPIO.output(EN, 0)
#time.sleep(0.1)
#GPIO.output(EN, 1)
