import numpy as np
from cv2 import cv2 as cv2
import time


img1 = cv2.imread('gsq.jpg', cv2.IMREAD_COLOR)
img1 = img1[100:600, 0:450]

# Fast
time_start = time.time()
fast = cv2.FastFeatureDetector_create()
kp_fast = fast.detect(img1, None)
time_end = time.time()
print('FAST:', time_end-time_start)

# 把特征点标记到图片上
img_fast = cv2.drawKeypoints(img1, kp_fast, None, (0, 0, 255), 4)
print('number of fast keypoints:', len(kp_fast))

# cv2.imshow('fast', img_fast)
# cv2.imshow('origin', img1)

final = np.zeros((500, 900, 3), np.uint8)
final[0:500, 0:450] = img1
final[0:500, 450:900] = img_fast
cv2.imshow('final', final)

cv2.waitKey(0)
cv2.destroyAllWindows()
