from uvc_camera import UVCCamera
import cv2
import time
import numpy as np
from transformation import *
from pymycobot import Mercury
import typing as T
from matplotlib.path import Path
from arm_control import *
from marker_utils import *
import stag

#ml = Mercury("/dev/ttyACM0")
ml = Mercury("/dev/ttyTHS0")

Tool_LEN = 175
Camera_LEN = 78
ml_camera_pos = [[-44.35, 36.52, 0.0, -17.03, 118.53, 56.09, 102.43],
[-63.87, -1.8, 0.0, -86.56, 101.7, 26.88, 70.13],
[-42.75, 28.72, 0.0, -97.87, 62.79, 55.46, 13.34],
[-32.46, 57.84, 0.0, -34.02, 89.35, 57.66, 55.98]]
tray_use_ids = [[3], [4], [5], [6]]
camera_data_all = []
cube_data = []
tray_data = []
cube_ids = []
tray_ids = []
tray_frame_plot = []
#cv vector to RotationMatrixT
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

def CvtEulerAngleToRotationMatrix(ptrEulerAngle):
    ptrSinAngle = np.sin(ptrEulerAngle)
    ptrCosAngle = np.cos(ptrEulerAngle)

    ptrRotationMatrix = np.zeros((3, 3))
    ptrRotationMatrix[0, 0] = ptrCosAngle[2] * ptrCosAngle[1]
    ptrRotationMatrix[0, 1] = ptrCosAngle[2] * ptrSinAngle[1] * ptrSinAngle[0] - ptrSinAngle[2] * ptrCosAngle[0]
    ptrRotationMatrix[0, 2] = ptrCosAngle[2] * ptrSinAngle[1] * ptrCosAngle[0] + ptrSinAngle[2] * ptrSinAngle[0]
    ptrRotationMatrix[1, 0] = ptrSinAngle[2] * ptrCosAngle[1]
    ptrRotationMatrix[1, 1] = ptrSinAngle[2] * ptrSinAngle[1] * ptrSinAngle[0] + ptrCosAngle[2] * ptrCosAngle[0]
    ptrRotationMatrix[1, 2] = ptrSinAngle[2] * ptrSinAngle[1] * ptrCosAngle[0] - ptrCosAngle[2] * ptrSinAngle[0]
    ptrRotationMatrix[2, 0] = -ptrSinAngle[1]
    ptrRotationMatrix[2, 1] = ptrCosAngle[1] * ptrSinAngle[0]
    ptrRotationMatrix[2, 2] = ptrCosAngle[1] * ptrCosAngle[0]

    return ptrRotationMatrix

#坐标转换为齐次变换矩阵，（x,y,z,rx,ry,rz）单位rad
def Transformation_matrix(coord):
    position_robot = coord[:3]
    pose_robot = coord[3:]
    RBT = CvtEulerAngleToRotationMatrix(pose_robot)
    PBT = np.array([[position_robot[0]],
                          [position_robot[1]],
                          [position_robot[2]]])
    temp = np.concatenate((RBT, PBT), axis=1)
    array_1x4 = np.array([[0, 0, 0, 1]])

    # 将两个数组按行拼接起来
    matrix = np.concatenate((temp, array_1x4), axis=0)
    return matrix


def Eyes_in_hand(coord, camera):
    #相机坐标
    Position_Camera = np.transpose(camera[:3])
    pose_camera = camera[3:]

    #机械臂坐标矩阵
    Matrix_BT = Transformation_matrix(coord)
    #手眼矩阵
    Matrix_TC = np.array([[0, -1, 0, Camera_LEN],
                        [1, 0, 0, 0],
                        [0, 0, 1, -Tool_LEN],
                        [0, 0, 0, 1]])
    #物体坐标（相机系）
    Position_Camera = np.append(Position_Camera, 1)
    
    Position_B = Matrix_BT @ Matrix_TC @ Position_Camera
    print("fact point",Position_B)
    return Position_B


def waitl():
	time.sleep(0.2)
	while (ml.is_moving()):
		time.sleep(0.03)

#recognition coord
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
	  
	  target_coords = np.array([tvec[0], tvec[1], tvec[2], Euler[0], Euler[1], Euler[2]])
	  # print("target_coords", target_coords)
	return target_coords
	
if __name__ == "__main__":
	np.set_printoptions(suppress=True, formatter={'float_kind':'{:.2f}'.format})

	camera_params = np.load("src/camera_params.npz")	#camera parameters
	mtx, dist = camera_params["mtx"], camera_params["dist"]

	MARKER_SIZE = 32	#aruco size mm
	#camera = UVCCamera(4, mtx, dist)  #randomly transforms between 0 and 4
	#camera.capture()	#get camera
	ml.set_gripper_mode(0)
	ml.set_tool_reference([0,0,Tool_LEN,0,0,0])
	ml.set_end_type(1)
	ml.set_gripper_value(0,100)
	time.sleep(1)
	#origin
	sp = 40
	#origin_anglesL = [-44.24, 15.56, 0.0, -102.59, 65.28, 52.06, 23.49] #regcognition attitude close to zero
	##origin_anglesL = [44.24, 15.56, 0.0, -102.59, -65.28, 52.06, -23.49] #regcognition attitude close to zero
	for ml_carmera_angles in ml_camera_pos:
	    camera = UVCCamera(4, mtx, dist)
	    camera.capture()
	    ml.send_angles(ml_carmera_angles, sp)
	    waitl()
	    camera.update_frame()
	    frame = camera.color_frame()
	    cur_coords = np.array(ml.get_base_coords())
	    (corners, ids, rejected_corners) = stag.detectMarkers(frame, 11)
	    print("ids", ids)
	    for i in range(len(ids)):
	        camera_data_all.append({"ids": ids[i], "cur_coords": cur_coords, "corners": corners[i]})
	#print(ml_camera_data_all)
	for data in camera_data_all:
	    if data['ids'] not in cube_ids and data['ids'] not in tray_use_ids:
	        cube_ids.append(data['ids'])
	        cube_data.append(data)
	    elif data['ids'] not in tray_ids and data['ids'] in tray_use_ids:
	        tray_ids.append(data['ids'])
	        tray_data.append(data)
	print("cube_data", cube_data)
	print("tray_data", tray_data)
	for tray in tray_data:
	    p = Path(tray["corners"])
	    tray_center_plot = p.vertices.mean(axis=0)
	    print(tray_center_plot)
	    
	exit()
	#recog
	
	 # type: ignore
	for ml_data in cube_data:
	    print("ids", ml_data["ids"])
	    print("corners", ml_data["corners"])
	    marker_pos_pack = calc_markers_base_position(ml_data["corners"], ml_data["ids"], MARKER_SIZE, mtx, dist)
	    #fact_bcl: calculate aruco coords in Base coodinate
	    # cur_coords = np.array(ml.get_base_coords())
	    print("mark", marker_pos_pack)	
	    recog_co = marker_pos_pack
	    cur_coords = ml_data["cur_coords"]
	    cur_bcl = cur_coords.copy()
	    cur_bcl[-3:] *= (np.pi/180)
	    print("current_coords", cur_bcl)
	    fact_bcl = Eyes_in_hand(cur_bcl, recog_co)
	    print("fact_bcl", fact_bcl)
	    #exit()
	
	    #attitude rotation origin pose
	    #only rotate rz
	    print("r", recog_co)
	    offset = (-recog_co[5] + cur_bcl[5]) * 180 / np.pi
	    #it can be rotated directly because the initial regcognition attitude is close to zer
	    #ml.send_base_coord(6, offset, 70)
	    print(offset)
	    waitl()
	
	    target_coords = cur_coords.copy()
	    target_coords[0] = fact_bcl[0]
	    target_coords[1] = fact_bcl[1]
	    target_coords[2] = fact_bcl[2] + 50
	    target_coords[5] = offset
	
	    print("target", target_coords)
	    # exit()
	    ml.send_base_coords(target_coords, sp)
	    waitl()
	    #continue
	    ml.set_gripper_value(100,100)
	    ml.send_base_coord(3, fact_bcl[2] - 50, sp)
	    waitl()
	    ml.set_gripper_value(20,100)
	    time.sleep(2)
	    ml.send_base_coord(3, fact_bcl[2] + 50 , sp)
	    waitl()
	    ml.send_base_coord(1, fact_bcl[0] + 50 , sp)
	    waitl()
	    ml.send_base_coord(3, fact_bcl[2] - 50, sp)
	    waitl()
	    ml.set_gripper_value(100,100)
	    ml.send_base_coord(3, fact_bcl[2] + 50 , sp)
	    waitl()
	    
	

	
	
	
	
	

