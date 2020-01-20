import numpy as np
from cv2 import cv2
import camera_configs
import time

imgL = cv2.imread('left/000000.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('right/000000.png', cv2.IMREAD_GRAYSCALE)
imgQ = cv2.imread('left/000015.png', cv2.IMREAD_GRAYSCALE)

# 得到点的3d坐标
num = 0
blockSize = 9
stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
disparity = stereo.compute(imgL, imgR)
threeD = cv2.reprojectImageTo3D(
    disparity.astype(np.float32)/16., camera_configs.Q)

# 匹配
time_start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
kpQ, des1 = sift.detectAndCompute(imgQ, None)
kpL, des2 = sift.detectAndCompute(imgL, None)
time_end = time.time()
print('SIFT extracting:', time_end-time_start)
ratio = 0.7

time_start = time.time()
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=10)
time_end = time.time()
print('SIFT matching:', time_end-time_start)
# matches = sorted(matches, key=lambda x: x.distance)

print('number of matches:', len(matches))
good = []
# ratio test as per Lowe's paper
matchesMask = [[0, 0] for i in range(len(matches))]
goodMatchCount = 0
for m in matches:
    # and m.distance < 300:
    if m[0].distance < ratio*m[1].distance and threeD[int(kpL[m[0].trainIdx].pt[1])][int(kpL[m[0].trainIdx].pt[0])][2] > 0:
        good.append(m[0])
        # print(m.distance)
        goodMatchCount += 1

print('number of good matches:', goodMatchCount)

# homograph test

# 获取关仝点的坐标

point2d = np.float32([kpQ[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
point3d = np.float32([threeD[int(kpL[m.trainIdx].pt[1])]
                      [int(kpL[m.trainIdx].pt[0])] for m in good]).reshape(-1, 1, 3)

found, rvec, tvec, inlier = cv2.solvePnPRansac(
    point3d, point2d, camera_configs.left_camera_matrix, None, reprojectionError=2)
rotM = cv2.Rodrigues(rvec)[0]
camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
print(camera_postion.T)
print(tvec)
print("number of inlier:", len(inlier))

final_match = []
for m in inlier:
    final_match.append(good[m[0]])

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   #    matchesMask=matchesMask,
                   flags=2)

img3 = cv2.drawMatches(imgQ, kpQ, imgL, kpL, final_match, None, **draw_params)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
