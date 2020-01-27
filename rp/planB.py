from cv2 import cv2
import numpy as np
import cameraRoboCar


def GetRootSIFT(descs):
    eps = 1e-7
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)
    return descs


def RedrawMatch(img, match):
    for m in match:
        xq = int(kpQ[m.queryIdx].pt[0])
        yq = int(kpQ[m.queryIdx].pt[1])
        xr = int(kpR[m.trainIdx].pt[0])+1024
        yr = int(kpR[m.trainIdx].pt[1])
        img = cv2.line(img, (xq, yq), (xr, yr), color=(
            0, 255, 0), thickness=3)
    return img


# read keypoints
data = np.loadtxt('superData.txt')
data = data.reshape(-1, 1, 5)
print(data.shape)

kpQ = []
kpQ_3d = []
for d in data:
    kpQ.append(cv2.KeyPoint(d[0][0], d[0][1], 1))
    kpQ_3d.append(d[0][2:5])


# feature matching
imgQ = cv2.imread('robocar/reference.jpg', cv2.IMREAD_COLOR)
imgR = cv2.imread('robocar/night.jpg', cv2.IMREAD_COLOR)
R_ref = np.mat([[0.2984, 0.9529, -0.0540],
                [-0.1656, 0.1074, 0.9803],
                [0.9400, -0.2836, 0.1899]])
t_ref = np.mat([17.432, -16.4696, 95.5569]).T
# print(R_ref)
# print(t_ref)
# print(R_ref*t_ref)

sift = cv2.xfeatures2d.SIFT_create()
kpQ, desQ = sift.compute(imgQ, kpQ)
kpR, desR = sift.detectAndCompute(imgR, None)

desQ = GetRootSIFT(desQ)
desR = GetRootSIFT(desR)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(desQ, desR, k=100)

print('number of matches:', len(matches))
good = []
good_all = []
ratio = 0.7
# ratio test as per Lowe's paper
matchesMask = [[0, 0] for i in range(len(matches))]
for m in matches:
    temp_nice = []
    # if m[0].distance < ratio*m[1].distance:
    #     good.append(m[0])
    for n in m:
        kpq = kpQ[n.queryIdx].pt
        kpr = kpR[n.trainIdx].pt
        pos_p2c = R_ref*np.mat(kpQ_3d[n.queryIdx]).T+t_ref
        search_radius = 5/pos_p2c[2]*400
        distance = (kpq[0]-kpr[0])*(kpq[0]-kpr[0]) + \
            (kpq[1]-kpr[1])*(kpq[1]-kpr[1])
        if distance < search_radius*search_radius:
            temp_nice.append(n)
        howlong = len(temp_nice)
    # if len(temp_nice) > 1:
    #     print(len(temp_nice))
    if (len(temp_nice) == 1) or ((len(temp_nice) > 1) and (temp_nice[0].distance < ratio*temp_nice[1].distance)):
        good.append(temp_nice[0])
        good_all.append(temp_nice)

print('number of good matches:', len(good))

src_pts = np.float32([kpQ_3d[m.queryIdx] for m in good]).reshape(-1, 1, 3)
dst_pts = np.float32([kpR[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
retval, rvec, tvec, inliers = cv2.solvePnPRansac(src_pts, dst_pts,
                                                 cameraRoboCar.rear_camera_matrix, None, reprojectionError=5)
# plot
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   #    matchesMask=matchesMask,
                   flags=2)


img_super = cv2.drawMatches(imgQ, kpQ, imgR, kpR,
                            good, None, **draw_params)

cv2.namedWindow('super_result', cv2.WINDOW_NORMAL)
cv2.imshow('super_result', img_super)


draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   #    matchesMask=matchesMask,
                   flags=2)
refine = []
refine.append(good[14])
refine.append(good[38])
refine.append(good[52])
# for i in range(len(good)-1):

#     pos_p2c = R_ref*np.mat(kpQ_3d[good[i].queryIdx]).T+t_ref
#     search_radius = 5/pos_p2c[2]*400
#     imgR_temp = imgR.copy()
#     imgR_temp = cv2.circle(imgR_temp, (int(kpQ[good[i].queryIdx].pt[0]),
#                                        int(kpQ[good[i].queryIdx].pt[1])), search_radius, color=(0, 0, 255), thickness=2)
#     img4 = cv2.drawMatchesKnn(imgQ, kpQ, imgR_temp, kpR,
#                               good_all[i:i+1], None, **draw_params)
#     print('index', i)
#     cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
#     cv2.imshow('result2', img4)

#     key = cv2.waitKey(0)
#     if key == ord("q"):
#         break
imgR_temp = imgR.copy()
for i in range(len(refine)):

    pos_p2c = R_ref*np.mat(kpQ_3d[refine[i].queryIdx]).T+t_ref
    search_radius = 5/pos_p2c[2]*400
    imgR_temp = cv2.circle(imgR_temp, (int(kpQ[refine[i].queryIdx].pt[0]),
                                       int(kpQ[refine[i].queryIdx].pt[1])), search_radius, color=(0, 0, 255), thickness=5)
    img4 = cv2.drawMatches(imgQ, kpQ, imgR_temp, kpR,
                           refine, None, **draw_params)

    img4 = RedrawMatch(img4, refine)

    print('index', i)
    cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
    cv2.imshow('result2', img4)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break


cv2.waitKey(0)

cv2.destroyAllWindows()

# test draw kepoint
# for m in kpQ:
#     img3 = cv2.circle(
#         imgQ, (int(m.pt[0]), int(m.pt[1])), 5, color=(0, 255, 0), thickness=-1)

# cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# cv2.imshow('result', img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
