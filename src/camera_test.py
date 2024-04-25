from uvc_camera import UVCCamera
import cv2
import time
import numpy as np
from transformation import *
from pymycobot import Mercury
import typing as T

from arm_control import *
from marker_utils import *
import stag

def rotvector2rot(rotvector):
    Rm = cv2.Rodrigues(rotvector)[0]
    return Rm
    
def CvtRotationMatrixToEulerAngle(pdtRotationMatrix):
    pdtEulerAngle = np.zeros(3)

    pdtEulerAngle[2] = np.arctan2(pdtRotationMatrix[1, 0], pdtRotationMatrix[0, 0])

    fCosRoll = np.cos(pdtEulerAngle[2])
    fSinRoll = np.sin(pdtEulerAngle[2])

    pdtEulerAngle[1] = np.arctan2(-pdtRotationMatrix[2, 0], (fCosRoll * pdtRotationMatrix[0, 0]) + (fSinRoll * pdtRotationMatrix[1, 0]))
    pdtEulerAngle[0] = np.arctan2((fSinRoll * pdtRotationMatrix[0, 2]) - (fCosRoll * pdtRotationMatrix[1, 2]), (-fSinRoll * pdtRotationMatrix[0, 1]) + (fCosRoll * pdtRotationMatrix[1, 1]))

    return pdtEulerAngle

def calc_markers_base_position(corners : NDArray, ids : T.List , marker_size : int, mtx : NDArray, dist : NDArray) -> T.List:
	if len(corners) == 0:
          return []

	rvecs, tvecs = solve_marker_pnp(corners, marker_size, mtx, dist)
	res = []
	for i, tvec, rvec in zip(ids, tvecs, rvecs):
	  tvec = tvec.squeeze().tolist()
	  rvec = rvec.squeeze().tolist()

	  rotvector = np.array([[rvec[0], rvec[1], rvec[2]]])
	  Rotation = rotvector2rot(rotvector)
	  Euler = CvtRotationMatrixToEulerAngle(Rotation)
	  #print(Rotation)
	  #print(Euler)
	  cam_coords = tvec + rvec
	  target_coords = cam_coords
	return target_coords
    
from math import cos, sin, pi
if __name__ == "__main__":
	np.set_printoptions(suppress=True, formatter={'float_kind':'{:.2f}'.format})

	camera_params = np.load("src/camera_params.npz")
	mtx, dist = camera_params["mtx"], camera_params["dist"]

	MARKER_SIZE = 32
	camera = UVCCamera(2, mtx, dist)
	camera.capture()
	while 1:
		camera.update_frame()
		frame = camera.color_frame()

		(corners, ids, rejected_corners) = stag.detectMarkers(frame, 11) # type: ignore
		print("ids", ids)
		marker_pos_pack = calc_markers_base_position(corners, ids, MARKER_SIZE, mtx, dist)
		print("marker_pos_pack", marker_pos_pack)

		cv2.imshow("calibed", frame)
		cv2.waitKey(1)


