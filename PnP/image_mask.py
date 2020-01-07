from cv2 import cv2 as cv2
import numpy as np
import time

img = cv2.imread('test1.jpeg', cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(img, (5, 5), 1)
m, n, d = img.shape
print(m)
print(n)
print(d)

# 䕭定蓝色的侷值
lower_blue = np.array([40, 150, 165])
upper_blue = np.array([150, 255, 250])
# 根据侷值构建掩模
mask = cv2.inRange(img, lower_blue, upper_blue)
left = cv2.bitwise_and(img, img, mask=mask)
mask = cv2.medianBlur(mask, 5)


# max_row = 0
# max_col = 0
# min_row = m
# min_col = n
# time_start = time.time()
# for i in range(0, m):
#     for j in range(0, n):
#         if mask[i][j] == 255:
#             if i > max_row:
#                 max_row = i
#             if j > max_col:
#                 max_col = j
#             if i < min_row:
#                 min_row = i
#             if j < min_col:
#                 min_col = j
# print("row", min_row, " ", max_row)
# print("col", min_col, " ", max_col)
# time_end = time.time()
# print('time 1:', time_end-time_start)


time_start = time.time()
x, y = mask.nonzero()
min_row = min(x)
max_row = max(x)
min_col = min(y)
max_col = max(y)

print("row", min_row, " ", max_row)
print("col", min_col, " ", max_col)
time_end = time.time()
print('time 2:', time_end-time_start)

cv2.circle(img, (min_col, min_row), 30, (0, 0, 255), -1)
cv2.circle(img, (min_col, max_row), 30, (0, 0, 255), -1)
cv2.circle(img, (max_col, min_row), 30, (0, 0, 255), -1)
cv2.circle(img, (max_col, max_row), 30, (0, 0, 255), -1)


cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
cv2.namedWindow('final', cv2.WINDOW_NORMAL)
final = np.hstack((img, left))
cv2.imshow('mask', mask)
cv2.imshow('origin', img)
cv2.imshow('final', final)
cv2.waitKey(0)

# def calBatchSum(img, x, y, batch_size):
#     half_size = batch_size / 2
#     sum = 0
#     for i in range(-half_size, half_size+1):
#         for j in range(-half_size, half_size+1):
#             sum += img[x+i][x+j]
#     return
