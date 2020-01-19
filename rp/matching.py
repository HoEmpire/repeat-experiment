import numpy as np
from cv2 import cv2 as cv2
import time

img1 = cv2.imread('000000.png', cv2.IMREAD_COLOR)
#img1 = cv2.imread('yld_query.jpeg', cv2.IMREAD_COLOR)
img2 = cv2.imread('000014.png', cv2.IMREAD_COLOR)

time_start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
time_end = time.time()
print('SIFT extracting:', time_end-time_start)
ratio = 0.7

time_start = time.time()
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)
time_end = time.time()
print('SIFT matching:', time_end-time_start)
# matches = sorted(matches, key=lambda x: x.distance)

print('number of matches:', len(matches))
good = []
# ratio test as per Lowe's paper
matchesMask = [[0, 0] for i in range(len(matches))]
goodMatchCount = 0
for m, n in matches:
    # and m.distance < 300:
    if m.distance < ratio*n.distance and kp1[m.queryIdx].pt[1] > 120 and kp2[m.trainIdx].pt[1] > 60:
        good.append(m)
        print(m.distance)
        goodMatchCount += 1
print('number of good matches:', goodMatchCount)

# homograph test

# 获取关仝点的坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
matchesMask = mask.ravel().tolist()

goodMatchCount = 0
for m in matchesMask:
    if m == 1:
        goodMatchCount += 1
print('number of good matches after homograph test:', goodMatchCount)

# plot
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   # matchesMask=matchesMask,
                   flags=0)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', img3)
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
