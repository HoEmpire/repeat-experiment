import numpy as np
from cv2 import cv2 as cv2
import time


def GetRootSIFT(descs):
    eps = 1e-7
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)
    return descs


def RedrawMatch(img, match):
    for m in match:
        xq = int(kp1[m.queryIdx].pt[0])
        yq = int(kp1[m.queryIdx].pt[1])
        xr = int(kp2[m.trainIdx].pt[0])+1024
        yr = int(kp2[m.trainIdx].pt[1])
        img = cv2.line(img, (xq, yq), (xr, yr), color=(
            255, 0, 0), thickness=3)
    return img


def RedrawRightMatch(img, match, id):
    m = match[id]
    xq = int(kp1[m.queryIdx].pt[0])
    yq = int(kp1[m.queryIdx].pt[1])
    xr = int(kp2[m.trainIdx].pt[0])+1024
    yr = int(kp2[m.trainIdx].pt[1])
    img = cv2.line(img, (xq, yq), (xr, yr), color=(
        0, 255, 0), thickness=3)
    return img


img1 = cv2.imread('robocar/reference.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('robocar/night.jpg', cv2.IMREAD_COLOR)

time_start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
des1 = GetRootSIFT(des1)
des2 = GetRootSIFT(des2)

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

    temp_nice = []
    for n in m:
        kpq = kp1[n.queryIdx].pt
        kpr = kp2[n.trainIdx].pt
        if (kpq[0]-kpr[0])*(kpq[0]-kpr[0])+(kpq[1]-kpr[1])*(kpq[1]-kpr[1]) < 1000:
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
draw_params = dict(matchColor=(255, 0, 0),
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


# for i in range(len(nice)-1):

#     img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2,
#                               nice_all[i:i+1], None, **draw_params)
#     img4 = cv2.drawMatches(img1, kp1, img2, kp2,
#                            nice[i:i+1], None, **draw_params_nice)
#     print('match ID:', i)
#     print(img3.shape)
#     print(img2.shape)
#     img3 = RedrawMatch(img3, nice_all[i])

#     cv2.namedWindow('result', cv2.WINDOW_NORMAL)
#     cv2.imshow('result', img3)

#     cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
#     cv2.imshow('result2', img4)

#     key = cv2.waitKey(0)
#     if key == ord("q"):
#         break

test_id = 199
right_id = 9
for i in range(0, 50):

    nice_all_temp = nice_all.copy()
    # nice_all_temp[test_id] = nice_all[test_id].copy()

    print(len(nice_all_temp[test_id]))
    pop_record = []
    for n in nice_all_temp[test_id]:
        kpq = kp1[n.queryIdx].pt

        kpr = kp2[n.trainIdx].pt
        distance = (kpq[0]-kpr[0])*(kpq[0]-kpr[0]) + \
            (kpq[1]-kpr[1])*(kpq[1]-kpr[1])
        print(distance)
        if distance < i*20*i*20:
            pop_record.append(n)
    nice_all_temp[test_id] = pop_record

    imgR_temp = img2.copy()
    imgR_temp = cv2.circle(imgR_temp, (int(kpq[0]),
                                       int(kpq[1])), i*20, color=(0, 0, 255), thickness=5)
    print(len(nice_all_temp[test_id]))
    img3 = cv2.drawMatchesKnn(img1, kp1, imgR_temp, kp2,
                              nice_all_temp[test_id:test_id+1], None, **draw_params)

    print('match ID:', i)

    img3 = RedrawMatch(img3, nice_all_temp[test_id])
    img3 = RedrawRightMatch(img3, nice_all[test_id], right_id)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', img3)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
