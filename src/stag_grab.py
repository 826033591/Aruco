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


def init_arm(arm : Mercury):
    arm.send_angles([-29.96, 57.82, 0.0, -63.23, 73.5, 64.58, -64.84], 50)
    time.sleep(5)

    arm.set_gripper_value(100, 50)
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
    
    mat_chain = [arm_coords, [75,0,0,0,0,90]]
    rot = arm_coords[-3:]
    p_base = homo_transform_point(mat_chain, pos)
    new_coords = np.concatenate([p_base, rot])
    return new_coords

def calc_target_coords(arm_coords, cam_coords) -> T.List:    
    assert len(arm_coords) == 6
    assert len(cam_coords) == 6
    
    hand_to_eye = [75, 10, 0,0,0,90]
    coords_base = homo_transform_frame([arm_coords, hand_to_eye, cam_coords])
    return coords_base


def calc_xy_move_coords(
    arm_coords: T.List, target_move: T.List
) -> T.List:
    z_now = arm_coords[2]
    new_coords = target_move.copy()
    new_coords[2] = z_now
    return new_coords

def calc_xy_move_coords_without_rot(
    arm_coords: T.List, target_move: T.List
) -> T.List:
    z_now = arm_coords[2]
    new_coords = target_move.copy()
    new_coords[2] = z_now
    new_coords[3:] = arm_coords[3:]
    return new_coords


def calc_markers_base_position(corners : NDArray, ids : T.List ,arm_base_coords : T.List, marker_size : int, mtx : NDArray, dist : NDArray) -> T.List:
    if len(corners) == 0:
        return []
    assert len(arm_base_coords) == 6
    
    rvecs, tvecs = solve_marker_pnp(corners, marker_size, mtx, dist)
    res = []
    for i, tvec, rvec in zip(ids, tvecs, rvecs):
        tvec = tvec.squeeze().tolist()
        rvec = rvec.squeeze().tolist()
        cam_coords = tvec + rvec
        target_coords = calc_target_coords(arm_base_coords, cam_coords)
        res.append((i, target_coords))
    return res

def move_to_item_and_grab(arm, now_base_coords ,target_base_coords):
    """
    从观察位置移动到抓取位置，执行抓取
    """
    target_base_coords[3:] = now_base_coords[3:]
    xy_coords = calc_xy_move_coords(now_base_coords, target_base_coords)
    # xy_coords[2] -= 50
    
    print(f"xy_coords : {xy_coords}")
    arm.send_base_coords(xy_coords, 50)
    time.sleep(3)

    target_base_coords[2] += 50 - 10 + 100
    print(f"target_coords : {target_base_coords}")
    
    arm.send_base_coords(target_base_coords, 50)
    time.sleep(4)
    
    while arm.is_moving() != 0:
        time.sleep(1)
    
    arm.set_gripper_value(10, 50)
    time.sleep(1)


def place_item_to_tray(arm):
    arm.send_base_coord(3, 280, 80)
    time.sleep(3)
    
    arm.send_base_coord(2, 50, 80)
    time.sleep(3)

    arm.send_base_coord(1, 180, 80)
    time.sleep(5)
    
    arm.send_base_coord(6, -90, 80)
    time.sleep(10)
    
    arm.send_base_coord(3, 50, 100)
    time.sleep(10)
    
    arm.set_gripper_value(20, 50)
    time.sleep(1)

    arm.send_base_coord(3, 350, 80)
    time.sleep(5)
    
    arm.send_base_coord(6, 0, 80)
    time.sleep(10)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:.2f}'.format})

    arm = Mercury("/dev/ttyTHS0")

    camera_params = np.load("src/camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]

    MARKER_SIZE = 32
    camera = UVCCamera(0, mtx, dist)
    camera.capture()

    while True:
        camera.update_frame()
        frame = camera.color_frame()
        if frame is None:
            continue
        camera.release()
        
        cv2.imshow("preview0", frame)
        cv2.waitKey(0)
        
        (corners, ids, rejected_corners) = stag.detectMarkers(frame, 11) # type: ignore
        if len(ids) != 0:
            now_base_coords = get_base_coords(arm)
            if now_base_coords is None:
                continue
            
            marker_pos_pack = calc_markers_base_position(corners, ids, now_base_coords, MARKER_SIZE, mtx, dist)
            marker_id, marker_base_coords = marker_pos_pack[0]

            print(f"marker in base frame : {marker_base_coords}")
            print(f"now_coords : {now_base_coords}")
            
        else:
            continue
        
        camera.capture()
        camera.update_frame()
        frame = camera.color_frame()
        if frame is None:
            continue
        
        cv2.imshow("preview1", frame)
        cv2.waitKey(0)
        
        (corners, ids, rejected_corners) = stag.detectMarkers(frame, 11) # type: ignore
        if len(ids) != 0:
            now_base_coords = get_base_coords(arm)
            if now_base_coords is None:
                continue
            
            marker_pos_pack = calc_markers_base_position(corners, ids, now_base_coords, MARKER_SIZE, mtx, dist)
            marker_id, marker_base_coords = marker_pos_pack[0]

        
    camera.release()
    cv2.destroyAllWindows()
