from uvc_camera import UVCCamera
import cv2
import time
from aruco_detector import ArucoDetector
from transformation import homo_transform_matrix
import numpy as np
import stag

camera_params = np.load("src/camera_params_rot.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]

MARKER_SIZE = 32

camera = UVCCamera(2, mtx, dist)
aruco_detector = ArucoDetector(mtx, dist, MARKER_SIZE)

camera.capture()

while True:
    camera.update_frame()
    frame = camera.color_frame()
    if frame is None:
        continue

    (corners, ids, rejected_corners) = stag.detectMarkers(frame, 11) # type: ignore
    if ids is not None:
        aruco_frame = frame.copy()
        rvecs, tvecs = aruco_detector.estimatePoseSingleMarkers(corners)
        aruco_detector.draw_marker(aruco_frame, corners, tvecs, rvecs, ids)
        marker_infos = aruco_detector.make_structure_data(corners, ids, rvecs, tvecs)
        aruco_detector.console_view_marker_pos3d(marker_infos)
        aruco_detector.draw_position_info(aruco_frame, corners, tvecs)
        cv2.imshow("aruco", aruco_frame)

    cv2.imshow("calibed", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
