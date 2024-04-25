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

import mercury_ros_api

def init_arm(arm):
    arm.send_angles([-29.96, 57.82, 0.0, -63.23, 73.5, 64.58, -64.84], 50)
    time.sleep(5)

    pump_off(arm)
    time.sleep(0.5)

    arm.set_tool_reference([0, 0, 70, 0, 0, 0])
    time.sleep(0.05)
    arm.set_end_type(1)



def calc_new_coords(arm_coords : T.List, pos : T.List) -> T.Union[np.ndarray, None]:
    """
    将相机坐标系下的点, 转换到机械臂基坐标系下, 姿态直接沿用末端姿态
    """
    # 单独算坐标
    if arm_coords is None or len(arm_coords) != 6:
        return None
    
    mat_chain = [arm_coords, [75, 10, 0,0,0,90]]
    rot = arm_coords[-3:]
    p_base = homo_transform_point(mat_chain, pos)
    new_coords = np.concatenate([p_base, rot])
    return new_coords

# def calc_new_coords(arm_coords, tvecs) -> T.Union[np.ndarray, None]:
#     # 单独算坐标
#     if arm_coords is None or len(arm_coords) != 6:
#         return None

#     mat = homo_transform_matrix(*arm_coords) @ homo_transform_matrix(75, 10, 0, 0, 0, 90)
#     rot = arm_coords[-3:]
#     p_end = np.vstack([np.reshape(tvecs, (3, 1)), 1])
#     p_base = np.squeeze((mat @ p_end)[:-1]).astype(int)
#     new_coords = np.concatenate([p_base, rot])
#     return new_coords


def calc_target_coords(arm_coords, tvec, rvec) -> T.Union[np.ndarray, None]:
    # 坐标+角度
    if arm_coords is None or len(arm_coords) != 6:
        return None
    tvec = tvec.squeeze().tolist()
    rvec = rvec.squeeze().tolist()
    mat = (
        homo_transform_matrix(*arm_coords)
        @ homo_transform_matrix(0, 70, 0, 0, 0, 90)
        @ homo_transform_matrix(*tvec, *rvec)
    )
    rot_mat = mat[:3, :3]
    r: np.ndarray = cvt_rotation_matrix_to_euler_angle(rot_mat)
    r = r * 180 / np.pi
    t: np.ndarray = mat[:3, 3]
    res = np.hstack([t.squeeze(), r.squeeze()])
    return res


def calc_xy_move_coords(
    arm_coords: T.List, target_move: T.List
) -> T.List:
    z_now = arm_coords[2]
    new_coords = target_move.copy()
    new_coords[2] = z_now
    return new_coords

def place_item_to_tray(arm):
    arm.send_base_coord(3, 280, 80)
    time.sleep(3)
    
    arm.send_base_coord(2, 50, 80)
    time.sleep(3)
    
    arm.send_base_coord(1, 180, 80)
    time.sleep(5)
    
    arm.send_base_coord(6, 90, 80)
    time.sleep(4)
    
    arm.send_base_coord(3, 50, 80)
    time.sleep(5)

    pump_off(arm)
    time.sleep(1)

    arm.send_base_coord(3, 350, 80)
    time.sleep(5)

def calc_markers_base_position(corners : NDArray, ids : T.List ,arm_base_coords : T.List, marker_size : int, mtx : NDArray, dist : NDArray) -> T.List:
    if len(corners) == 0:
        return []
    assert len(arm_base_coords) == 6
    
    rvecs, tvecs = solve_marker_pnp(corners, marker_size, mtx, dist)
    res = []
    for i, tvec, rvec in zip(ids, tvecs, rvecs):
        target_coords = calc_target_coords(arm_base_coords, tvec, rvec)
        res.append((i, target_coords))
    return res

def move_to_item_and_grab(arm, now_base_coords ,target_base_coords):
    xy_coords = calc_xy_move_coords(now_base_coords, target_base_coords)
    xy_coords[2] -= 50
    
    print(f"xy_coords : {xy_coords}")
    arm.send_base_coords(xy_coords, 50)
    time.sleep(3)

    target_base_coords[2] += 50 - 7
    print(f"target_coords : {target_base_coords}")
    
    arm.send_base_coords(target_base_coords, 50)
    time.sleep(4)
    
    pump_on(arm)
    time.sleep(1)

def aruco_arm():
    while True:
        camera.update_frame()
        frame = camera.color_frame()
        if frame is None:
            continue
        
        cv2.imshow("preview", frame)
        cv2.waitKey(0)
        
        (corners, ids, rejected_corners) = stag.detectMarkers(frame, 11) # type: ignore
        if len(ids) != 0:
            detect_frame = frame.copy()
            rvecs, tvecs = solve_marker_pnp(corners, MARKER_SIZE, mtx, dist)
            draw_marker(detect_frame, corners, tvecs, rvecs, ids, mtx, dist)
            
            now_base_coords = get_base_coords(arm)
            if now_base_coords is None:
                continue
            
            target_base_coords = calc_new_coords(now_base_coords, tvecs[0])

            print(f"marker in camera frame : {tvecs}")
            print(f"marker in base frame : {target_base_coords}")
            print(f"now_coords : {now_base_coords}")
            
            move_to_item_and_grab(arm, now_base_coords, target_base_coords)

            place_item_to_tray(arm)
            
            arm.send_angles([-29.96, 57.82, 0.0, -63.23, 73.5, 64.58, -64.84], 50)
            time.sleep(5)

            camera.capture()
            break
    
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    arm = Mercury("/dev/ttyTHS0")
    # init_arm(arm)
    camera_params = np.load("src/camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]

    MARKER_SIZE = 32
    camera = UVCCamera(0, mtx, dist)
    camera.capture()

    base = mercury_ros_api.MapNavigation()

    # 1. Fix a position first, and then send a forward command
    # 2. Perform the stag code capture of the robotic arm
    # 3. Send a backward rotation command after the scraping is completed
    # 4. Send forward commands
    # 5. Perform the stag code capture of the robotic arm
    # 6. Send a backward rotation command after the grabbing is completed

    # while True:
    # base.goStraight(0.2,1)

    # aruco_arm()

    base.goBack(0.2,0.5)

    # base.turnRight(0.5,5)

    # base.goStraight(0.2,0.5)

    # aruco_arm()

    # base.goBack(0.2,0.5)

    # base.turnRight(0.5,0.5)



            


