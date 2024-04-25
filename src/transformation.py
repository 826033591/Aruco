import numpy as np
from numpy.typing import NDArray
import typing as T
import cv2


def Rx(theta) -> NDArray:
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def Ry(theta) -> NDArray:
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def Rz(theta) -> NDArray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def degree2radian(degree):
    return (degree / 180) * np.pi


def rotation_matrix(rx, ry, rz, order="ZYX") -> NDArray:
    order = order.upper()
    if len(order) != 3 or set(order) != set("XYZ"):
        raise Exception("Order must be string of component of XYZ or xyz")
    mat = np.identity(3)
    rx = degree2radian(rx)
    ry = degree2radian(ry)
    rz = degree2radian(rz)
    for c in order:
        if c == "X":
            mat = mat @ Rx(rx)
        elif c == "Y":
            mat = mat @ Ry(ry)
        elif c == "Z":
            mat = mat @ Rz(rz)
    return mat


def homo_transform_matrix(x, y, z, rx, ry, rz, order="ZYX") -> NDArray:
    rot_mat = rotation_matrix(rx, ry, rz, order=order)
    trans_vec = np.array([[x, y, z, 1]]).T
    mat = np.vstack([rot_mat, np.zeros((1, 3))])
    mat = np.hstack([mat, trans_vec])
    return mat


def get_angle_from_rect(corners: NDArray) -> float:
    center, size, angle = cv2.minAreaRect(corners)
    return angle



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

def homo_transform_point(matrix_chain : T.List[T.List], pos : T.List) -> T.List:
    """homo transform one point from rear of matrix chain to the front matrix chain frame.

    Args:
        matrix_chain (T.List[T.List]): size [N, 6], a list of consecutive convert chain
        pos (T.List): a point in end of matrix chain frame.

    Returns:
        T.List: the point in first of matrix chain frame.
    """
    if len(matrix_chain) == 0:
        return pos
    
    mat = np.identity(4, dtype=np.float64)
    for m in matrix_chain:
        assert len(m) == 6
        mat = mat @ homo_transform_matrix(*m)
    
    p_end = np.vstack([np.reshape(pos, (3, 1)), 1])
    p_base = np.squeeze((mat @ p_end)[:-1]).astype(int).tolist()
    assert len(p_base) == 3
    return p_base

def homo_transform_frame(matrix_chain : T.List[T.List]) -> T.List:
    """homo transform one point from rear of matrix chain to the front matrix chain frame.

    Args:
        matrix_chain (T.List[T.List]): size [N, 6], a list of consecutive convert chain
        pos (T.List): a point in end of matrix chain frame.

    Returns:
        T.List: the point in first of matrix chain frame.
    """
    mat = np.identity(4, dtype=np.float64)
    
    for m in matrix_chain:
        assert len(m) == 6
        mat = mat @ homo_transform_matrix(*m)

    rot = cvt_rotation_matrix_to_euler_angle(mat[:3, :3]).squeeze()
    rot = np.rad2deg(rot)
    p_base = mat[:3,3].squeeze()
    res = np.hstack([p_base, rot]).tolist()
    return res

def euler_to_base_vec(rx, ry, rz) -> T.Tuple[NDArray, NDArray, NDArray]:
    # 定义初始单位向量
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # 定义ZYX欧拉角（弧度）
    rx, ry, rz = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)

    # 计算旋转矩阵
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])

    # 计算ZYX旋转矩阵
    R_zyx = R_z @ R_y @ R_x

    # 计算基向量
    x_new = np.dot(R_zyx, x_axis)
    y_new = np.dot(R_zyx, y_axis)
    z_new = np.dot(R_zyx, z_axis)

    return (x_new,y_new,z_new)


if __name__ == "__main__":
    corners = np.array([(0, 40), (40, 40), (0, 0), (40, 0)])
    get_angle_from_rect(corners)
