from cv2 import cv2 as cv2
import numpy as np
img = cv2.imread('pic1.jpg', 0)
img = cv2.medianBlur(img, 5)
img = cv2.medianBlur(img, 5)
img3 = img[1500:2000, 2200:2800]
cimg = cv2.imread('pic1.jpg', cv2.IMREAD_COLOR)
circles = cv2.HoughCircles(img3, cv2.HOUGH_GRADIENT, 1, 200,
                           param1=50, param2=50, minRadius=100, maxRadius=2000)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img3, (i[0], i[1]), i[2], (0, 255, 0), 5)
    # draw the center of the circle
    cv2.circle(img3, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)
cv2.imshow('detected circles', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
