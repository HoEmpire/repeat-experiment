from cv2 import cv2
import numpy as np
import math
object_3d_points = np.array(([0, 0, 0],
                             [0, 200, 0],
                             [150, 0, 0],
                             [150, 200, 0]), dtype=np.double)
object_2d_point = np.array(([2985, 1688],
                            [5081, 1690],
                            [2997, 2797],
                            [5544, 2757]), dtype=np.double)
camera_matrix = np.array(([6800.7, 0, 3065.8],
                          [0, 6798.1, 1667.6],
                          [0, 0, 1.0]), dtype=np.double)
dist_coefs = np.array([0.0234, 0.5209, 0, 0], dtype=np.double)
# 求解相机位姿
found, rvec, tvec = cv2.solvePnP(
    object_3d_points, object_2d_point, camera_matrix, None)
rotM = cv2.Rodrigues(rvec)[0]
camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
print(camera_postion.T)

object_3d_points = np.array(([0, 0, 0],
                             [75, 0, 0],
                             [0, 0, 75],
                             [75, 0, 75]), dtype=np.double)
# object_2d_point = np.array(([944, 1611],
#                             [1474, 2138],
#                             [944, 2138],
#                             [1474, 1611]), dtype=np.double)
object_2d_point = np.array(([1611, 944],
                            [2138, 944],
                            [1611, 1611],
                            [2138, 1611]), dtype=np.double)
camera_matrix = np.array(([2936.8, 0, 1658.0],
                          [0, 2931.6, 1242.5],
                          [0, 0, 1.0]), dtype=np.double)
dist_coefs = np.array([0.0234, 0.5209, 0, 0], dtype=np.double)
# 求解相机位姿
found, rvec, tvec = cv2.solvePnP(
    object_3d_points, object_2d_point, camera_matrix, None)
rotM = cv2.Rodrigues(rvec)[0]
camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
print(camera_postion.T)
