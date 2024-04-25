from uvc_camera import UVCCamera
import cv2
import numpy as np

cam = UVCCamera(0)
cam.capture()

for i in range(50):
    cam.update_frame()

frame = cam.color_frame()
# frame = np.rot90(frame, 3)
cv2.imwrite("p.jpg", frame)