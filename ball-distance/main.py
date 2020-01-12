from cv2 import cv2 as cv2
import numpy as np
import time
import math


def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1,     0,         0],
                    [0,     math.cos(theta[0]), -math.sin(theta[0])],
                    [0,     math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]),  0,   math.sin(theta[1])],
                    [0,           1,   0],
                    [-math.sin(theta[1]),  0,   math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]),  -math.sin(theta[2]),  0],
                    [math.sin(theta[2]),  math.cos(theta[2]),   0],
                    [0,           0,           1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


img = cv2.imread('pic4.jpg', cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(img, (5, 5), 0)
img2 = img.copy()

# extract ROI by color threshold
lower_blue = np.array([30, 180, 150])
upper_blue = np.array([100, 220, 200])

# mask the ROI
mask = cv2.inRange(img, lower_blue, upper_blue)
left = cv2.bitwise_and(img, img, mask=mask)
image, contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_record = 0
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area > 2000:
        print("find the size with enough area")
        if max_record < area:
            max_record = area
            result = contours[i]


x, y, w, h = cv2.boundingRect(result)
xc = round((2*x+w)/2)
yc = round((2*y+h)/2)

x_start = xc-2*w
x_end = xc+2*w
y_start = yc-1*h
y_end = yc+3*h

img_ROI_origin = img.copy()
# img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 10)
# img = cv2.circle(img, (xc, yc), 20, (0, 0, 255), -1)
img_ROI_origin = cv2.rectangle(
    img_ROI_origin, (x_start, y_start), (x_end, y_end), (0, 0, 255), 30)
cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
cv2.imshow('ROI', img_ROI_origin)
cv2.waitKey(0)

img_ROI = img[y_start:y_end, x_start:x_end]
img_ROI = cv2.cvtColor(img_ROI, cv2.COLOR_BGR2GRAY)
img_ROI = cv2.medianBlur(img_ROI, 5)
img_ROI = cv2.medianBlur(img_ROI, 5)
# img_ROI = cv2.medianBlur(img_ROI, 5)
# img_ROI = cv2.medianBlur(img_ROI, 5)
circles = cv2.HoughCircles(img_ROI, cv2.HOUGH_GRADIENT, 1, 200,
                           param1=40, param2=40, minRadius=30, maxRadius=500)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img_ROI, (i[0], i[1]), i[2], (0, 255, 0), 5)
    # draw the center of the circle
    cv2.circle(img_ROI, (i[0], i[1]), 2, (0, 0, 255), 3)
    # draw the outer circle
    cv2.circle(img, (i[0]+x_start, i[1]+y_start), i[2], (0, 0, 255), 20)
    # draw the center of the circle
    cv2.circle(img, (i[0]+x_start, i[1]+y_start), 20, (0, 0, 255), -1)
    print("圆心", (i[0]+x_start, i[1]+y_start))
    cc_x = i[0]+x_start
    cc_y = i[1]+y_start

cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)
cv2.imshow('detected circles', img_ROI)

# # solve PnP
angle_H2G = (0, 46.4/180*math.pi, 0)
t_H2G = [0, 0, 475]
R_H2G = eulerAnglesToRotationMatrix(angle_H2G)
# print(R_H2G)
R_C2H = np.mat(([0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]), dtype=np.double)
camera_matrix = np.mat(([2936.8, 0, 1658.0],
                        [0, 2931.6, 1242.5],
                        [0, 0, 1.0]), dtype=np.double)
ball_2d_point = np.mat([cc_x, cc_y, 1], dtype=np.double)
ball_2d_point = ball_2d_point.reshape(3, 1)
result1 = np.linalg.inv(camera_matrix)*ball_2d_point
result2 = R_H2G*R_C2H*result1
z = t_H2G[2]/result2[2]
ball_pos = int(-z[0])*result2
print(ball_pos)
# object_3d_points = np.array(([37.5, -37.5, 0],
#                              [37.5, 37.5, 0],
#                              [-37.5, 37.5, 0],
#                              [-37.5, -37.5, 0
#                               ]), dtype=np.double)
# object_2d_point = np.array(corner_refine, dtype=np.double)

# # object_3d_points = np.array(([0, 0, 0],
# #                              [0, 0, -75],
# #                              [0, 75, -75],
# #                              [0, 75, 0
# #                               ]), dtype=np.double)
# # object_3d_points = np.array(([37.5, -37.5, 0],
# #                              [37.5, 37.5, 0],
# #                              [-37.5, 37.5, 0]), dtype=np.double)
# # object_2d_point = np.array(corner_refine[0:3], dtype=np.double)


# camera_matrix = np.array(([2936.8, 0, 1658.0],
#                           [0, 2931.6, 1242.5],
#                           [0, 0, 1.0]), dtype=np.double)
# dist_coefs = np.array([0.0234, 0.5209, 0, 0], dtype=np.double)

# # calculate camera pose
# found, rvec, tvec = cv2.solvePnP(
#     object_3d_points, object_2d_point, camera_matrix, dist_coefs)
# rotM = cv2.Rodrigues(rvec)[0]
# camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
# print('相机系下标志的位置：', tvec.T)
# print('全局系下相机的位置：', camera_postion.T)
# rotM = rotM.T
# # print(rotM)
# roll = math.atan2(rotM[2, 1], rotM[2, 2])/math.pi*180
# pitch = math.atan2(-rotM[2, 0], -math.sqrt(rotM[2, 1]
#                                            * rotM[2, 1]+rotM[2, 2]*rotM[2, 2]))/math.pi*180
# yaw = math.atan2(rotM[1, 0], rotM[0, 0])/math.pi*180
# print('朝向角：', pitch)

# # show the result
# for c in corner_refine:
#     cv2.circle(img, (int(c[0]), int(c[1])), 30, (0, 0, 255), -1)


# cv2.namedWindow('first step', cv2.WINDOW_NORMAL)
# first = np.hstack((img2, left))

# cv2.namedWindow('after', cv2.WINDOW_NORMAL)

# cv2.imshow('first step', first)
# cv2.imshow('after', img)

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
cv2.imshow('mask', mask)
cv2.imshow('origin', img)
cv2.waitKey(0)
