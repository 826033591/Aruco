#!/bin/bash

# 488 PY.04 G23 G0
# 487 PY.03 G24 EN
echo 488 > /sys/class/gpio/export
echo out > /sys/class/gpio/PY.04/direction

echo 487 > /sys/class/gpio/export
echo out > /sys/class/gpio/PY.03/direction


echo 1 > /sys/class/gpio/PY.03/value
sleep 0.1
echo 0 > /sys/class/gpio/PY.03/value
