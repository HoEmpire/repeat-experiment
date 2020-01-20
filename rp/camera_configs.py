# filename: camera_configs.py
from cv2 import cv2
import numpy as np

left_camera_matrix = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
                               [0.000000000000e+00, 7.188560000000e+02,
                                   1.852157000000e+02],
                               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
left_distortion = None


right_camera_matrix = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
                                [0.000000000000e+00, 7.188560000000e+02,
                                    1.852157000000e+02],
                                [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
right_distortion = None

R = np.eye(3, 3)
T = np.array([-3.861448000000e+02, 0, 0])  # 平移关系向量

size = (1241, 376)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
