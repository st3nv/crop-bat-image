import cv2
import numpy as np
import imutils

image = cv2.imread("C://Users//yzgua//Desktop//bat//test_large//720.jpg")
cv2.namedWindow('original', 0)
cv2.imshow("original", image)

(h,w) = image.shape[:2]

(cX, cY) = (w//2, h//2)

M = cv2.getRotationMatrix2D((cX, cY), -1.75, 1.1)
rotated = cv2.warpAffine(image, M, (w, h))

# rotated = imutils.rotate(image, -1.75)

cv2.imwrite("C:\\Users\\yzgua\\Desktop\\bat\\pic\\rot\\720_rot.jpg", rotated)


# rotated = imutils.rotate(image, 180)
cv2.imshow("Rotated", rotated)

cv2.waitKey(0)