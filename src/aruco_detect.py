import time

import cv2
import cv2.aruco as aruco
import numpy as np

from Aruco.src.marker_utils import solve_marker_pnp

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})
camera_params = np.load("camera_params.npz")  # camera parameters
mtx, dist = camera_params["mtx"], camera_params["dist"]


def CvtRotationMatrixToEulerAngle(pdtRotationMatrix):
    pdtEulerAngle = np.zeros(3)

    pdtEulerAngle[2] = np.arctan2(pdtRotationMatrix[1, 0], pdtRotationMatrix[0, 0])

    fCosRoll = np.cos(pdtEulerAngle[2])
    fSinRoll = np.sin(pdtEulerAngle[2])

    pdtEulerAngle[1] = np.arctan2(-pdtRotationMatrix[2, 0],
                                  (fCosRoll * pdtRotationMatrix[0, 0]) + (fSinRoll * pdtRotationMatrix[1, 0]))
    pdtEulerAngle[0] = np.arctan2((fSinRoll * pdtRotationMatrix[0, 2]) - (fCosRoll * pdtRotationMatrix[1, 2]),
                                  (-fSinRoll * pdtRotationMatrix[0, 1]) + (fCosRoll * pdtRotationMatrix[1, 1]))

    return pdtEulerAngle


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                          aruco_dict,
                                                          parameters=parameters)
    print(corners)
    rvecs, tvecs = solve_marker_pnp(corners, 22, mtx, dist)
    for tvec, rvec in zip(rvecs, tvecs):
        tvec = tvec.squeeze().tolist()
        rvec = rvec.squeeze().tolist()
        rotvector = np.array([[rvec[0], rvec[1], rvec[2]]])
        print("rotvector", rotvector)
        Rotation = cv2.Rodrigues(rotvector)[0]
        Euler = CvtRotationMatrixToEulerAngle(Rotation)
        target_coords = np.array([tvec[0], tvec[1], tvec[2], Euler[0] * 180 / np.pi, Euler[1] * 180 / np.pi, Euler[2] * 180 / np.pi])
        # print(corners)
        # print("tvec", tvec)
        # print("rvec", rvec)
        # print("rotvector", rotvector)
        # print("Rotation", Rotation)
        # print("Euler", Euler)
        print("target_coords", target_coords)
        print("=======")

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    # time.sleep(2)
    # s = input()

