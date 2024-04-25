from uvc_camera import UVCCamera
import cv2
from transformation import *
from pymycobot import Mercury
import typing as T
from matplotlib.path import Path
from arm_control import *
from marker_utils import *
import stag
from math import cos, sin, pi
import rospy
import time
import actionlib
import sys
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import *
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist

Tool_LEN = 175
Camera_LEN = 78
sp = 40
MARKER_SIZE = 32  # aruco size mm
ml_camera_pos_a = [[-44.35, 36.52, 0.0, -17.03, 118.53, 56.09, 102.43],
                   [-63.87, -1.8, 0.0, -86.56, 101.7, 26.88, 70.13],
                   [-42.75, 28.72, 0.0, -97.87, 62.79, 55.46, 13.34],
                   [-32.46, 57.84, 0.0, -34.02, 89.35, 57.66, 55.98]]

mr_camera_pos_a = [[49.62, 11.46, 0.0, -60.17, -110.25, 43.66, -91.08],
                   [67.66, -5.67, 0.0, -89.52, -104.43, 23.09, -79.66],
                   [47.47, 10.58, 0.0, -106.06, -64.27, 50.15, -27.61],
                   [32.37, 43.67, 0.0, -39.45, -93.44, 59.02, -71.64]]
ml_camera_pos_b = [[-41.88, 30.62, -1.14, -102.57, 59.89, 58.11, 9.63]]
mr_camera_pos_b = [[45.94, 25.33, 0.15, -102.31, -58.65, 55.8, -13.94]]

goal_1 = [(1.8811798181533813, 1.25142673254013062, 0.9141818042023212, 0.4053043657122249)]  # A
goal_2 = [(2.1266400814056396, 0.398377299, 0.37314402, 0.9277734297)]  # to b
goal_3 = [(3.942732810974, 2.20178937, 0.36272676664, 0.9318955374722)]  # B
tray_use_ids = [[3], [4], [5], [6]]
camera_data_a = []
camera_data_b = []

mr = Mercury("/dev/ttyACM0")
ml = Mercury("/dev/ttyTHS0")
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})
camera_params = np.load("src/camera_params.npz")  # camera parameters
mtx, dist = camera_params["mtx"], camera_params["dist"]


class MapNavigation:
    def __init__(self):
        self.goalReached = None
        rospy.init_node('map_navigation', anonymous=False)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub_setpose = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.pub_cancel = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)

    # amcl init
    def set_pose(self, position_x, position_y, orientation_z, orientation_w, covariance):
        pose = PoseWithCovarianceStamped()
        pose.header.seq = 0
        pose.header.stamp.secs = 0
        pose.header.stamp.nsecs = 0
        pose.header.frame_id = 'map'
        pose.pose.pose.position.x = position_x
        pose.pose.pose.position.y = position_y
        pose.pose.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, 1.57)
        pose.pose.pose.orientation.x = 0.0
        pose.pose.pose.orientation.y = 0.0
        pose.pose.pose.orientation.z = orientation_z
        pose.pose.pose.orientation.w = orientation_w
        pose.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, covariance]
        rospy.sleep(1)
        self.pub_setpose.publish(pose)
        rospy.loginfo('Published robot pose: %s' % pose)

    # move_base
    def moveToGoal(self, xGoal, yGoal, orientation_z, orientation_w):
        ac = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while (not ac.wait_for_server(rospy.Duration.from_sec(5.0))):
            sys.exit(0)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position = Point(xGoal, yGoal, 0)
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = orientation_z
        goal.target_pose.pose.orientation.w = orientation_w

        rospy.loginfo("Sending goal location ...")
        ac.send_goal(goal)

        ac.wait_for_result(rospy.Duration(60))

        if (ac.get_state() == GoalStatus.SUCCEEDED):
            rospy.loginfo("You have reached the destination")
            return True
        else:
            rospy.loginfo("The robot failed to reach the destination")
            return False

    def shutdown(self):
        rospy.loginfo("Quit program")
        rospy.sleep()

    def pub_vel(self, x, y, theta):
        twist = Twist()
        twist.linear.x = x
        twist.linear.y = y
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = theta
        self.pub.publish(twist)


# cv vector to RotationMatrixT
def rotvector2rot(rotvector):
    Rm = cv2.Rodrigues(rotvector)[0]
    return Rm


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


# 坐标转换为齐次变换矩阵，（x,y,z,rx,ry,rz）单位rad
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


def Eyes_in_hand(coord, camera, arm):
    # 相机坐标
    Position_Camera = np.transpose(camera[:3])
    pose_camera = camera[3:]

    # 机械臂坐标矩阵
    Matrix_BT = Transformation_matrix(coord)
    # 手眼矩阵
    if arm == "left":
        Matrix_TC = np.array([[0, -1, 0, Camera_LEN],
                              [1, 0, 0, -0],
                              [0, 0, 1, -Tool_LEN],
                              [0, 0, 0, 1]])
    else:
        Matrix_TC = np.array([[0, 1, 0, Camera_LEN - 4],
                              [-1, 0, 0, 25],
                              [0, 0, 1, -Tool_LEN - 7],
                              [0, 0, 0, 1]])
    # 物体坐标（相机系）
    Position_Camera = np.append(Position_Camera, 1)

    Position_B = Matrix_BT @ Matrix_TC @ Position_Camera
    # print("fact point",Position_B)
    return Position_B


def waitl():
    time.sleep(0.2)
    while (ml.is_moving()) or (mr.is_moving()):
        time.sleep(0.03)


# recognition coord
def calc_markers_base_position(corners: NDArray, ids: T.List, marker_size: int, mtx: NDArray, dist: NDArray) -> T.List:
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


def initialize_robot():
    for mc in [ml, mr]:
        if mc == ml:
            z = Tool_LEN
            origin_angles = [-42.75, 28.72, 0.0, -97.87, 62.79, 55.46, 13.34]
        else:
            z = Tool_LEN + 7
            origin_angles = [54.99, 0.24, 0.0, -102.79, -72.01, 38.43, -41.91]
        mc.set_gripper_mode(0)
        mc.set_tool_reference([0, 0, z, 0, 0, 0])
        mc.set_end_type(1)
        mc.set_gripper_value(100, 100)
        time.sleep(2)
        mc.send_angles(origin_angles, sp)
        waitl()
        mc.set_gripper_value(0, 100)
        time.sleep(1)
    map_navigation = MapNavigation()
    return map_navigation


def detect_objects(arm, camera_pos, camera_data):
    if arm == "left":
        mc = ml
        camera_id = 4
    else:
        mc = mr
        camera_id = 0
    for carmera_angles in camera_pos:
        camera = UVCCamera(camera_id, mtx, dist)
        camera.capture()
        mc.send_angles(carmera_angles, sp)
        waitl()
        camera.update_frame()
        frame = camera.color_frame()
        cur_coords = np.array(mc.get_base_coords())
        (corners, ids, rejected_corners) = stag.detectMarkers(frame, 11)
        print("ids", ids)
        for i in range(len(ids)):
            camera_data.append({"ids": ids[i], "cur_coords": cur_coords, "corners": corners[i], "arm": arm})


def judge_tray_cube(camera_data):
    cube_data = []
    tray_data = []
    cube_ids = []
    tray_ids = []
    for data in camera_data:
        if data['ids'] not in cube_ids and data['ids'] not in tray_use_ids:
            cube_ids.append(data['ids'])
            cube_data.append(data)
        elif data['ids'] not in tray_ids and data['ids'] in tray_use_ids:
            tray_ids.append(data['ids'])
            tray_data.append(data)
    return cube_data, tray_data


def get_obj_coords(corners, ids, cur_coords, arm):
    marker_pos_pack = calc_markers_base_position(corners, ids, MARKER_SIZE, mtx, dist)
    recog_co = marker_pos_pack
    cur_bcl = cur_coords.copy()
    cur_bcl[-3:] *= (np.pi / 180)
    if arm == "left":
        fact_bcl = Eyes_in_hand(cur_bcl, recog_co, "left")
    else:
        fact_bcl = Eyes_in_hand(cur_bcl, recog_co, "right")
    offset = (-recog_co[5] + cur_bcl[5]) * 180 / np.pi
    return offset, fact_bcl


def get_tray_plots(tray_data):
    def get_all_plots(offset, fact_bcl):
        point = fact_bcl.copy()
        px, py, pz = point[0], point[1], point[2]
        p0 = [px, py]
        af = (offset + 90) * pi / 180
        Hei = 115
        Len = 175
        px1 = px + Hei * cos(af)
        py1 = py + Hei * sin(af)
        p1 = [px1, py1]
        px2 = px + Len * sin(af)
        py2 = py - Len * cos(af)
        p2 = [px2, py2]
        px3 = px + Hei * cos(af) + Len * sin(af)
        py3 = py + Hei * sin(af) - Len * cos(af)
        p3 = [px3, py3]
        return [p0, p1, p2, p3]
    tray_frame_plot = []
    for tray in tray_data:
        offset, fact_bcl = get_obj_coords(tray["corners"], tray["ids"], tray["cur_coords"], tray['arm'])
        if len(tray_data) != 4 and tray["ids"] == [4]:
            tray_frame_plot = get_all_plots(offset, fact_bcl)
        else:
            tray_frame_plot.append((fact_bcl[0], fact_bcl[1]))
    return tray_frame_plot


def catch_cube(mc, fact_bcl, offset):
    mc.send_base_coord(1, fact_bcl[0], sp)
    waitl()
    mc.send_base_coord(2, fact_bcl[1], sp)
    waitl()
    mc.send_base_coord(6, offset, sp)
    waitl()
    mc.set_gripper_value(70, 100)
    mc.send_base_coord(3, fact_bcl[2] - 50, sp)
    waitl()
    mc.set_gripper_value(20, 100)
    time.sleep(2)
    mc.send_base_coord(3, fact_bcl[2] + 50, sp)
    waitl()


def put_cube_in(cube_data, tray_data):
    def get_center_plot(tray_frame_plot):
        tray_frame = Path(tray_frame_plot)
        tray_frame_center_plot = tray_frame.vertices.mean(axis=0)
        return tray_frame, tray_frame_center_plot
    tray_frame_plot = get_tray_plots(tray_data)
    tray_frame, tray_frame_center_plot = get_center_plot(tray_frame_plot)
    for data in cube_data:
        offset, fact_bcl = get_obj_coords(data["corners"], data["ids"], data['cur_coords'], data["arm"])
        if tray_frame.contains_point((fact_bcl[0], fact_bcl[1])):
            continue
        # if fact_bcl[1] > 0:
        if data["arm"] == "left":
            mc = ml
            y = 50
            orgin_angles = [-39.53, 17.31, 0.0, -142.05, 52.94, 75.09, -11.68]
        else:
            mc = mr
            y = 0
            orgin_angles = [49.62, 11.46, 0.0, -60.17, -110.25, 43.66, -91.08]
        catch_cube(mc, fact_bcl, offset)
        mc.send_base_coord(1, tray_frame_center_plot[0], sp)
        waitl()
        mc.send_base_coord(2, tray_frame_center_plot[1] + y, sp)
        waitl()
        mc.send_base_coord(3, fact_bcl[2] - 50, sp)
        waitl()
        mc.set_gripper_value(70, 100)
        mc.send_base_coord(3, fact_bcl[2] + 50, sp)
        waitl()
        mc.send_angles(orgin_angles, sp)
        waitl()


def put_cube_out(cube_data):
    use_list = []
    for data in cube_data:
        offset, fact_bcl = get_obj_coords(data["corners"], data["ids"], data['cur_coords'], data["arm"])
        y = fact_bcl[1]
        use_list.append([y, offset, fact_bcl])
    if use_list[0][0] > use_list[1][0]:
        left_offset, left_bcl = use_list[0][1], use_list[0][2]
        right_offset, right_bcl = use_list[1][1], use_list[1][2]
    else:
        left_offset, left_bcl = use_list[1][1], use_list[1][2]
        right_offset, right_bcl = use_list[0][1], use_list[0][2]
    for mc in [mr, ml]:
        if mc == mr:
            orgin_angles = [49.62, 11.46, 0.0, -60.17, -110.25, 43.66, -91.08]
            y = -200
            bcl = right_bcl
            offset = right_offset
        else:
            orgin_angles = [-39.53, 17.31, 0.0, -142.05, 52.94, 75.09, -11.68]
            y = 200
            bcl = left_bcl
            offset = left_offset
        catch_cube(mc, bcl, offset)
        mc.send_base_coord(2, bcl[1] + y, sp)
        waitl()
        mc.send_base_coord(3, bcl[2] - 50, sp)
        waitl()
        mc.set_gripper_value(70, 100)
        mc.send_base_coord(3, bcl[2] + 50, sp)
        waitl()
        mc.send_angles(orgin_angles, sp)
        waitl()


def put_up_tray(tray_data):
    ml.set_gripper_value(60, 10)
    mr.set_gripper_value(60, 10)
    time.sleep(1)
    # print(tray_data)
    for tray in tray_data:
        offset, fact_bcl = get_obj_coords(tray["corners"], tray["ids"], tray["cur_coords"], tray['arm'])
        point = fact_bcl.copy()
        px, py, pz = point[0], point[1], point[2]
        if tray["ids"] == [4] and tray['arm'] == "left":
            af = (offset + 90) * pi / 180
            mc = ml
            Hei = 57
            Len = -20
        elif tray["ids"] == [5] and tray['arm'] == "right":
            af = (offset - 90) * pi / 180
            mc = mr
            Hei = 57
            Len = 30
        else:
            continue
        # p3
        px3 = px + Hei * cos(af) + Len * sin(af)
        py3 = py + Hei * sin(af) - Len * cos(af)
        pz3 = pz
        p3 = [px3, py3, pz3]
        mc.send_base_coord(6, offset, sp)
        mc.send_base_coord(2, py3, sp)
        mc.send_base_coord(1, px3, sp)
        mc.send_base_coord(3, pz - 10, sp)
    waitl()
    ml.set_gripper_value(0, 10)
    mr.set_gripper_value(0, 10)
    time.sleep(2)
    current_coordsl = ml.get_base_coords()
    current_coordsr = mr.get_base_coords()
    waitl()
    ml.send_base_coord(3, current_coordsl[2] + 20, 20)
    mr.send_base_coord(3, current_coordsr[2] + 20, 20)
    waitl()


def put_down_tray():
    current_coordsl = ml.get_base_coords()
    current_coordsr = mr.get_base_coords()
    ml.send_base_coord(3, current_coordsl[2], 20)
    mr.send_base_coord(3, current_coordsr[2], 20)
    waitl()
    ml.set_gripper_value(70, 10)
    mr.set_gripper_value(70, 10)
    time.sleep(2)


def car_move(map_navigation, flag):
    if flag == 0:
        goal_plot = goal_1
    else:
        goal_plot = goal_3
    x_goal, y_goal, orientation_z, orientation_w = goal_2[0]
    flag_feed_goalReached = map_navigation.moveToGoal(x_goal, y_goal, orientation_z, orientation_w)
    if flag_feed_goalReached:
        x_goal, y_goal, orientation_z, orientation_w = goal_plot[0]
        flag_feed_goalReached = map_navigation.moveToGoal(x_goal, y_goal, orientation_z, orientation_w)
        if flag_feed_goalReached:
            print("command completed")
        else:
            raise ValueError
    else:
        raise ValueError
    time.sleep(12)


def detect_objects_all(flag):
    if flag == 0:
        detect_objects("left", ml_camera_pos_a, camera_data_a)
        detect_objects("right", mr_camera_pos_a, camera_data_a)
        cube_data, tray_data = judge_tray_cube(camera_data_a)  # 获取盒子和方块相关数据
    else:
        ml.send_angles([-41.88, 30.62, -1.14, -102.57, 59.89, 58.11, 9.63], 10)
        mr.send_angles([45.94, 25.33, 0.15, -102.31, -58.65, 55.8, -13.94], 10)
        waitl()
        ml.set_gripper_value(0, 10)
        mr.set_gripper_value(0, 10)
        time.sleep(12)
        detect_objects("left", ml_camera_pos_b, camera_data_b)
        detect_objects("right", mr_camera_pos_b, camera_data_b)
        ml.set_gripper_value(50, 10)
        mr.set_gripper_value(50, 10)
        cube_data, tray_data = judge_tray_cube(camera_data_b)  # 获取盒子和方块相关数据
    return cube_data, tray_data


if __name__ == "__main__":
    map_navigation = initialize_robot()  # 初始化robot，返回小车导航对象
    car_move(map_navigation, 0)  # 小车先回到初始点位再移动到桌前准备夹取
    cube_data, tray_data = detect_objects_all(0)  # 双臂扫描,返回方块和盒子位置
    put_cube_in(cube_data, tray_data)  # 将木块夹入盒子
    put_up_tray(tray_data)  # 抬起盒子
    car_move(map_navigation, 1)  # 移动到另外一张桌子
    put_down_tray()  # 放下盒子
    cube_data, tray_data = detect_objects_all(1)  # 双臂扫描,返回方块和盒子位置
    put_cube_out(cube_data)  # 取出方块
