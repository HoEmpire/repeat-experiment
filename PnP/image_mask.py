from cv2 import cv2 as cv2
import numpy as np
import time

img = cv2.imread('img/pic1.jpeg', cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(img, (5, 5), 0)

# 䕭定蓝色的侷值
lower_blue = np.array([40, 140, 165])
upper_blue = np.array([180, 255, 250])
# lower_blue = np.array([40, 180, 165])
# upper_blue = np.array([150, 255, 220])
# 根据侷值构建掩模
mask = cv2.inRange(img, lower_blue, upper_blue)
left = cv2.bitwise_and(img, img, mask=mask)
image, contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area > 5000:
        print("find the size with enough area")
        result = contours[i]


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
    if batch_sum < 20:
        result_sum.append(batch_sum)
        corner.append([y, x])

corner_refine = []

head = 0
while head < len(corner)-1:
    tmp = []
    tmp.append(corner[head])
    for rear in range(1, len(corner)-head):
        if abs(corner[head][0]-corner[head+rear][0])+abs(corner[head][1]-corner[head+rear][1]) < 10:
            tmp.append(corner[head+rear])
        else:
            sum_x = 0
            sum_y = 0
            n = len(tmp)
            for t in tmp:
                sum_x += t[0]
                sum_y += t[1]
            corner_refine.append([sum_x/n, sum_y/n])
            break
    head += 1
print(corner_refine)
print(type(corner_refine))

# print("row", min_row, " ", max_row)
# print("col", min_col, " ", max_col)
# time_end = time.time()
# print('time 2:', time_end-time_start)

# cv2.circle(img, (min_col, min_row), 30, (0, 0, 255), -1)
# cv2.circle(img, (min_col, max_row), 30, (0, 0, 255), -1)
# cv2.circle(img, (max_col, min_row), 30, (0, 0, 255), -1)
# cv2.circle(img, (max_col, max_row), 30, (0, 0, 255), -1)

# cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
# cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
# cv2.namedWindow('final', cv2.WINDOW_NORMAL)
# final = np.hstack((img, left))
# cv2.imshow('mask', mask)
# cv2.imshow('origin', img)
# cv2.imshow('final', final)

# cv2.namedWindow('edge', cv2.WINDOW_NORMAL)
# cv2.imshow('edge', edge)

# cv2.namedWindow('step5', cv2.WINDOW_NORMAL)
# cv2.imshow('step5', step5)

# cv2.waitKey(0)

# def calBatchSum(img, x, y, batch_size):
#     half_size = batch_size / 2
#     sum = 0
#     for i in range(-half_size, half_size+1):
#         for j in range(-half_size, half_size+1):
#             sum += img[x+i][x+j]
#     return
