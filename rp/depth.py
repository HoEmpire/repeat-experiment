import numpy as np
from cv2 import cv2
import camera_configs


imgL = cv2.imread('left/000000.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('right/000000.png', cv2.IMREAD_GRAYSCALE)


cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 640, 0)
cv2.createTrackbar("num", "depth", 2, 10, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)


def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])
        print(blockSize)
        print(num)


cv2.setMouseCallback("depth", callbackFunc, None)


while True:

    # 调参用
    num = cv2.getTrackbarPos("num", "depth")
    blockSize = cv2.getTrackbarPos("blockSize", "depth")
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5
    # num = 4
    # blockSize = 11

    stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0,
                         beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    threeD = cv2.reprojectImageTo3D(
        disparity.astype(np.float32)/16., camera_configs.Q)

    cv2.imshow("left", imgL)
    cv2.imshow("right", imgR)
    cv2.imshow("depth", disp)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break


cv2.destroyAllWindows()
