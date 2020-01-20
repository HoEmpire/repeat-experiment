from cv2 import cv2
import numpy as np


def GetRootSIFT(descs):
    eps = 1e-7
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)
    return descs


data = np.loadtxt('images.txt')
data = data.reshape(-1, 1, 3)
print(data.shape)
print(len(data))
print(data[0][0][2])
# print(data[0] == -1)

kpQ = []
PointIndex = []
for i in range(len(data)):
    if data[i][0][2] != -1:
        kpQ.append(cv2.KeyPoint(data[i][0][0], data[i][0][1], 1))
        PointIndex.append(data[i][0][2])

imgQ = cv2.imread('robocar/reference.jpg', cv2.IMREAD_COLOR)
imgR = cv2.imread('robocar/night.jpg', cv2.IMREAD_COLOR)

sift = cv2.xfeatures2d.SIFT_create()
print(kpQ[0].pt)
kpQ, desQ = sift.compute(imgQ, kpQ)
print(kpQ[0].pt)
kpR, desR = sift.detectAndCompute(imgR, None)

desQ = GetRootSIFT(desQ)
desR = GetRootSIFT(desR)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(desQ, desR, k=2)

print('number of matches:', len(matches))
good = []
ratio = 0.8
# ratio test as per Lowe's paper
matchesMask = [[0, 0] for i in range(len(matches))]
for m in matches:
    # and m.distance < 300:
    if m[0].distance < ratio*m[1].distance:
        good.append(m[0])

print('number of good matches:', len(good))

# src_pts = np.float32([kpQ[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
# dst_pts = np.float32([kpR[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
# M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 10)
# matchesMask = mask.ravel().tolist()

# plot
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   #    matchesMask=matchesMask,
                   flags=2)


img_super = cv2.drawMatches(imgQ, kpQ, imgR, kpR,
                            good, None, **draw_params)

cv2.namedWindow('super_result', cv2.WINDOW_NORMAL)
cv2.imshow('super_result', img_super)
cv2.waitKey(0)

cv2.destroyAllWindows()

# test draw kepoint
# for m in kpQ:
#     img3 = cv2.circle(
#         img1, (int(m.pt[0]), int(m.pt[1])), 5, color=(0, 255, 0), thickness=-1)

# cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# cv2.imshow('result', img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
