import numpy as np
from cv2 import cv2 as cv2
import time


def GetRootSIFT(descs):
    eps = 1e-7
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)
    return descs


img1 = cv2.imread('robocar/reference2.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('robocar/night.jpg', cv2.IMREAD_COLOR)

time_start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

des1 = GetRootSIFT(des1)
des2 = GetRootSIFT(des2)

time_end = time.time()
print('SIFT extracting:', time_end-time_start)
ratio = 0.8

time_start = time.time()
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=100)
time_end = time.time()
print('SIFT matching:', time_end-time_start)
# matches = sorted(matches, key=lambda x: x.distance)

print('number of matches:', len(matches))
good = []
nice = []
nice_all = []
# ratio test as per Lowe's paper
matchesMask = [[0, 0] for i in range(len(matches))]
goodMatchCount = 0
for m in matches:
    # and m.distance < 300:
    if m[0].distance < ratio*m[1].distance:
        good.append(m)
        # print(m.distance)
        goodMatchCount += 1

    for n in m:
        temp_nice = []
        kpq = kp1[n.queryIdx].pt
        kpr = kp2[n.trainIdx].pt
        if (kpq[0]-kpr[0])*(kpq[0]-kpr[0])+(kpq[1]-kpr[1])*(kpq[1]-kpr[1]) < 900:
            temp_nice.append(n)
    if (len(temp_nice) == 1) or ((len(temp_nice) > 1) and (temp_nice[0].distance < ratio*temp_nice[1].distance)):
        nice.append(temp_nice[0])
        nice_all.append(m)

        # print(n.distance, '\n')

print('number of good matches:', goodMatchCount)
print('number of super matches:', len(nice))

src_pts = np.float32([kp1[m.queryIdx].pt for m in nice]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in nice]).reshape(-1, 1, 2)
M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 10)
# matchesMask = mask.ravel().tolist()

# plot
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   #    matchesMask=matchesMask,
                   flags=2)

draw_params_nice = dict(matchColor=(0, 0, 255),
                        singlePointColor=(255, 0, 0),
                        #    matchesMask=matchesMask,
                        flags=2)

# img_super = cv2.drawMatches(img1, kp1, img2, kp2,
#                             nice, None, **draw_params)

# cv2.namedWindow('super_result', cv2.WINDOW_NORMAL)
# cv2.imshow('super_result', img_super)
# cv2.waitKey(0)


for i in range(len(nice)-1):

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2,
                              nice_all[i:i+1], None, **draw_params)
    img4 = cv2.drawMatches(img1, kp1, img2, kp2,
                           nice[i:i+1], None, **draw_params_nice)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', img3)

    cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
    cv2.imshow('result2', img4)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
