import numpy as np
from pymycobot import Mercury

LEFT_ARM = 0
RIGHT_ARM = 1


def coords_degree2radian(coords: np.ndarray):
    _coords = coords.copy()
    _coords = _coords.astype(np.float64)
    _coords[-3:] = degree2radian(_coords[-3:])
    return _coords


def coords_radian2degree(coords: np.ndarray):
    _coords = coords.copy()
    _coords = _coords.astype(np.float64)
    _coords[-3:] = radian2degree(_coords[-3:])
    return _coords


def degree2radian(x):
    return x / 180 * np.pi


def radian2degree(x):
    return x * 180 / np.pi


def cvt_euler_angle_to_rotation_matrix(ptr_euler_angle):
    ptr_sin_angle = np.sin(ptr_euler_angle)
    ptr_cos_angle = np.cos(ptr_euler_angle)

    ptr_rotation_matrix = np.zeros((3, 3))
    ptr_rotation_matrix[0, 0] = ptr_cos_angle[2] * ptr_cos_angle[1]
    ptr_rotation_matrix[0, 1] = (
        ptr_cos_angle[2] * ptr_sin_angle[1] * ptr_sin_angle[0]
        - ptr_sin_angle[2] * ptr_cos_angle[0]
    )
    ptr_rotation_matrix[0, 2] = (
        ptr_cos_angle[2] * ptr_sin_angle[2] * ptr_cos_angle[0]
        + ptr_sin_angle[2] * ptr_sin_angle[0]
    )
    ptr_rotation_matrix[1, 0] = ptr_sin_angle[2] * ptr_cos_angle[1]
    ptr_rotation_matrix[1, 1] = (
        ptr_sin_angle[2] * ptr_sin_angle[1] * ptr_sin_angle[0]
        + ptr_cos_angle[2] * ptr_cos_angle[0]
    )
    ptr_rotation_matrix[1, 2] = (
        ptr_sin_angle[2] * ptr_sin_angle[1] * ptr_cos_angle[0]
        - ptr_cos_angle[2] * ptr_sin_angle[0]
    )
    ptr_rotation_matrix[2, 0] = -ptr_sin_angle[1]
    ptr_rotation_matrix[2, 1] = ptr_cos_angle[1] * ptr_sin_angle[0]
    ptr_rotation_matrix[2, 2] = ptr_cos_angle[1] * ptr_cos_angle[0]

    return ptr_rotation_matrix


def cvt_rotation_matrix_to_euler_angle(rotation_matrix):
    euler_angle = np.zeros(3)
    euler_angle[2] = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    f_cos_roll = np.cos(euler_angle[2])
    f_sin_roll = np.sin(euler_angle[2])

    euler_angle[1] = np.arctan2(
        -rotation_matrix[2, 0],
        (f_cos_roll * rotation_matrix[0, 0]) + (f_sin_roll * rotation_matrix[1, 0]),
    )
    euler_angle[0] = np.arctan2(
        (f_sin_roll * rotation_matrix[0, 2]) - (f_cos_roll * rotation_matrix[1, 2]),
        (-f_sin_roll * rotation_matrix[0, 1]) + (f_cos_roll * rotation_matrix[1, 1]),
    )

    return euler_angle


def vr_to_base(posture):
    position = np.array(posture[:3])
    rotation = np.array(posture[3:])
    matrix = cvt_euler_angle_to_rotation_matrix(rotation)
    T = np.vstack((np.hstack((matrix, position.reshape(3, 1))), np.array([0, 0, 0, 1])))

    TRB = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    Tp = np.dot(TRB, T)

    rotation_matrix = Tp[:3, :3]
    position = Tp[:3, 3]
    rotation = cvt_rotation_matrix_to_euler_angle(rotation_matrix)

    return np.hstack((position, rotation))


#######################################################################################################################
#
#                       双臂基坐标系转单臂坐标系 input:1.posture基坐标系位姿（x,y,z,rx,ry,rz)单位mm\rad
#                                                    2.dire(0:left  1:Right)
#                                             output:single单臂坐标系位姿
#
#
#######################################################################################################################
def base_to_single(posture, dire):
    position = np.array(posture[:3])
    rotation = np.array(posture[3:])
    matrix = cvt_euler_angle_to_rotation_matrix(rotation)
    T = np.vstack((np.hstack((matrix, position.reshape(3, 1))), np.array([0, 0, 0, 1])))
    L1 = 38.42  # 131.42 - 93
    # L2 = 310
    L2 = 0

    eccentricity = 31.97
    af = np.pi / 6
    TRB = np.array(
        [
            [np.cos(af), np.sin(af), 0, 27.6868],
            [0, 0, 1, -L2],
            [np.sin(af), -np.cos(af), 0, -22.435],
            [0, 0, 0, 1],
        ]
    )
    TLB = np.array(
        [
            [np.cos(af), -np.sin(af), 0, 27.6868],
            [0, 0, -1, L2],
            [np.sin(af), np.cos(af), 0, -22.435],
            [0, 0, 0, 1],
        ]
    )

    if dire == 0:
        Tp = np.dot(TLB, T)
    else:
        Tp = np.dot(TRB, T)

    rotation_matrix = Tp[:3, :3]
    position = Tp[:3, 3]
    rotation = cvt_rotation_matrix_to_euler_angle(rotation_matrix)

    return np.hstack((position, rotation))


#######################################################################################################################
#
#                       单臂坐标系转双臂基坐标系 input:1.posture单臂坐标系位姿（x,y,z,rx,ry,rz)单位mm\rad
#                                                    2.dire(0:left  1:Right)
#                                             output:Base双臂基坐标系位姿
#
#
#######################################################################################################################
def single_to_base(posture, dire):
    posture = np.array(posture)
    rotation = posture[3:]
    matrix = cvt_euler_angle_to_rotation_matrix(rotation)
    T = np.vstack(
        (np.hstack((matrix, posture[:3].reshape(3, 1))), np.array([0, 0, 0, 1]))
    )
    L1 = 38.42  # 131.42 - 93
    # L2 = 310
    L2 = 0
    eccentricity = 31.97
    af = np.pi / 6
    TBR = np.array(
        [
            [np.cos(af), 0, np.sin(af), L1 * np.sin(af) - eccentricity],
            [np.sin(af), 0, -np.cos(af), -L1 * np.cos(af)],
            [0, 1, 0, L2],
            [0, 0, 0, 1],
        ]
    )
    TBL = np.array(
        [
            [np.cos(af), 0, np.sin(af), L1 * np.sin(af) - eccentricity],
            [-np.sin(af), 0, np.cos(af), L1 * np.cos(af)],
            [0, -1, 0, L2],
            [0, 0, 0, 1],
        ]
    )
    if dire == 1:
        TBF = np.dot(TBR, T)
    else:
        TBF = np.dot(TBL, T)

    rotation_matrix = TBF[:3, :3]
    position = TBF[:3, 3]
    rotation = cvt_rotation_matrix_to_euler_angle(rotation_matrix)

    return np.hstack((position, rotation))


if __name__ == "__main__":
    mL = Mercury("/dev/ttyACM0")
    coff = 180 / np.pi
    inv_coff = np.pi / 180
    speed = 50

    coord = [1, 0, 0, 0, 0, 0]
    # new_coord = eye_to_base(coord)

    # def example():
    #     #single to base
    #     postureL = mL.get_coords()  #获取单臂位姿
    #     print(postureL)

    #     for i in range(3):
    #         postureL[i+3] *= inv_coff   #角度转弧度
    #     Base_postureL = single_to_base(postureL, LEFT_ARM)  #得到左臂BASE坐标

    #     #base to single
    #     Single_postureL = base_to_single(Base_postureL, LEFT_ARM)
    #     for i in range(3):
    #         Single_postureL[i+3] *= coff   #弧度转角度
    #     print(Single_postureL)
    #     # mL.send_coords(Single_postureL, speed, 1)
    # example()
