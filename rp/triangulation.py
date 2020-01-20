from cv2 import cv2
import numpy as np
import cameraRoboCar as camera


def GetRootSIFT(descs):
    eps = 1e-7
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)
    return descs


def get_color(depth):
    up_th = 50.0
    low_th = 10.0
    th_range = up_th - low_th
    depth = -depth*th_range
    print(depth)
    if depth > up_th:
        depth = up_th
    if depth < low_th:
        depth = low_th
    color = (0, int(255 * depth / th_range), 0)  #

    return color


img1 = cv2.imread('robocar/reference.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('robocar/reference4.jpg', cv2.IMREAD_COLOR)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

des1 = GetRootSIFT(des1)
des2 = GetRootSIFT(des2)

ratio = 0.7

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m in matches:
    # and m.distance < 300:
    if m[0].distance < ratio*m[1].distance:
        good.append(m[0])

src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findEssentialMat(
    src_pts, dst_pts, camera.rear_camera_matrix, cv2.RANSAC)

better = []
for i in range(len(mask)):
    if mask[i] == 1:
        better.append(good[i])


src_pts = np.float32([kp1[m.queryIdx].pt for m in better]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in better]).reshape(-1, 1, 2)
retval, R, t, mask = cv2.recoverPose(
    M, src_pts, dst_pts, camera.rear_camera_matrix)

best = []
for i in range(len(mask)):
    if mask[i] != 0:
        best.append(better[i])

src_pts = np.float32([kp1[m.queryIdx].pt for m in best]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in best]).reshape(-1, 1, 2)

print('number of good matches:', len(good))
print("number of good matches after RANSAC:", len(better))
print("number of good matches after recvoer pose:", len(best))
print(R)
print(t)
projMat = np.c_[R, t]
print(projMat, '\n')
point4D = cv2.triangulatePoints(np.eye(3, 4), projMat, src_pts, dst_pts)
print(point4D[:, 0:5])
print(point4D[2, 1]/point4D[3, 1])


for i in range(len(best)):
    color = get_color(
        float(point4D[2, i]/point4D[3, i]))
    # print(point4D[2, i]/point4D[3, i])
    img3 = cv2.circle(
        img1, (src_pts[i][0][0], src_pts[i][0][1]), 5, color, thickness=-1)

cv2.namedWindow('depth result', cv2.WINDOW_NORMAL)
cv2.imshow('depth result', img3)

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   #    matchesMask=matchesMask,
                   flags=2)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, best, None, **draw_params)
# cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# cv2.imshow('result', img3)
for i in range(len(best)-1):

    img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                           best[i:i+1], None, **draw_params)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', img3)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
