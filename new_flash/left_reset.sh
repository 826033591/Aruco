#!/bin/bash

# 453 PS.04 G4
# 440 PQ.05 G5
# python left_flash.py
echo 440 > /sys/class/gpio/export
echo out > /sys/class/gpio/PQ.05/direction

echo 453 > /sys/class/gpio/export
echo out > /sys/class/gpio/PS.04/direction


echo 1 > /sys/class/gpio/PQ.05/value
sleep 0.1
echo 0 > /sys/class/gpio/PQ.05/value
