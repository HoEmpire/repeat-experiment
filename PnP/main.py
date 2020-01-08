from cv2 import cv2 as cv2
import numpy as np
import time
import math
img = cv2.imread('img/pic5.jpeg', cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(img, (5, 5), 0)
img2 = img.copy()

# extract ROI by color threshold
lower_blue = np.array([40, 140, 165])
upper_blue = np.array([160, 255, 250])

# mask the ROI
mask = cv2.inRange(img, lower_blue, upper_blue)
left = cv2.bitwise_and(img, img, mask=mask)
image, contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area > 10000:
        print("find the size with enough area")
        result = contours[i]

# calculate batch grayscale value to get corner
batch_size = 7
half_size = int(batch_size / 2)
print('patch_size:', half_size*2+1)
result_sum = []
corner = []
i = 0
for k in result:
    x = k[0][1]
    y = k[0][0]
    batch_sum = 0
    for i in range(-half_size, half_size+1):
        for j in range(-half_size, half_size+1):
            if mask[x+i][y+j] == 255:
                batch_sum += 1
    if batch_sum < 22:
        result_sum.append(batch_sum)
        corner.append([y, x])

# group the coarse corner
corner_refine = []
head = 0
while head < len(corner)-1:
    tmp = []
    tmp.append(corner[head])
    for rear in range(1, len(corner)-head):
        if abs(corner[head][0]-corner[head+rear][0])+abs(corner[head][1]-corner[head+rear][1]) < 10:
            tmp.append(corner[head+rear])
        else:
            break
    sum_x = 0
    sum_y = 0
    n = len(tmp)
    for t in tmp:
        sum_x += t[0]
        sum_y += t[1]
    corner_refine.append([sum_x/n, sum_y/n])
    head = head+rear
if len(corner_refine) <= 3:
    corner_refine.append(corner[-1])
print(corner)
print(corner_refine)

# solve PnP
object_3d_points = np.array(([37.5, -37.5, 0],
                             [37.5, 37.5, 0],
                             [-37.5, 37.5, 0],
                             [-37.5, -37.5, 0
                              ]), dtype=np.double)
object_2d_point = np.array(corner_refine, dtype=np.double)

# object_3d_points = np.array(([0, 0, 0],
#                              [0, 0, -75],
#                              [0, 75, -75],
#                              [0, 75, 0
#                               ]), dtype=np.double)
# object_3d_points = np.array(([37.5, -37.5, 0],
#                              [37.5, 37.5, 0],
#                              [-37.5, 37.5, 0]), dtype=np.double)
# object_2d_point = np.array(corner_refine[0:3], dtype=np.double)


camera_matrix = np.array(([2936.8, 0, 1658.0],
                          [0, 2931.6, 1242.5],
                          [0, 0, 1.0]), dtype=np.double)
dist_coefs = np.array([0.0234, 0.5209, 0, 0], dtype=np.double)

# calculate camera pose
found, rvec, tvec = cv2.solvePnP(
    object_3d_points, object_2d_point, camera_matrix, dist_coefs)
rotM = cv2.Rodrigues(rvec)[0]
camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
print('相机系下标志的位置：', tvec.T)
print('全局系下相机的位置：', camera_postion.T)
rotM = rotM.T
# print(rotM)
roll = math.atan2(rotM[2, 1], rotM[2, 2])/math.pi*180
pitch = math.atan2(-rotM[2, 0], -math.sqrt(rotM[2, 1]
                                           * rotM[2, 1]+rotM[2, 2]*rotM[2, 2]))/math.pi*180
yaw = math.atan2(rotM[1, 0], rotM[0, 0])/math.pi*180
print('朝向角：', pitch)

# show the result
for c in corner_refine:
    cv2.circle(img, (int(c[0]), int(c[1])), 30, (0, 0, 255), -1)


cv2.namedWindow('first step', cv2.WINDOW_NORMAL)
first = np.hstack((img2, left))
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('after', cv2.WINDOW_NORMAL)
cv2.imshow('mask', mask)
cv2.imshow('first step', first)
cv2.imshow('after', img)

cv2.waitKey(0)
