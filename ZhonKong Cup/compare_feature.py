import numpy as np
from cv2 import cv2 as cv2
import time


img1 = cv2.imread('ball.jpeg', cv2.IMREAD_COLOR)

#cv2.namedWindow('amilia', cv2.WINDOW_NORMAL)

#cv2.namedWindow('saber', cv2.WINDOW_NORMAL)
#img = cv2.resize(img, (600, 400))

# Surf
time_start = time.time()
surf = cv2.xfeatures2d.SURF_create(200)
kp_surf, des_surf = surf.detectAndCompute(img1, None)
time_end = time.time()
print('SURF:', time_end-time_start)

# Sift
time_start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(img1, None)
time_end = time.time()
print('SIFT:', time_end-time_start)

# Fast
time_start = time.time()
fast = cv2.FastFeatureDetector_create()
kp_fast = fast.detect(img1, None)
time_end = time.time()
print('FAST:', time_end-time_start)

# 把特征点标记到图片上
img_surf = cv2.drawKeypoints(img1, kp_surf, None, (0, 0, 255), 4)
img_sift = cv2.drawKeypoints(img1, kp_sift, None, (0, 0, 255), 4)
img_fast = cv2.drawKeypoints(img1, kp_fast, None, (0, 0, 255), 4)

#cv2.imshow('saber', img2)
cv2.imshow('amilia_surf', img_surf)
cv2.imshow('amilia_sift', img_sift)
cv2.imshow('amilia_fast', img_fast)
cv2.waitKey(0)
cv2.destroyAllWindows()
