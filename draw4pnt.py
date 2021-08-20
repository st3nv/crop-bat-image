import cv2
import os
import numpy as np
import math

path = r'C:\Users\yzgua\Desktop\bat\erode'
filename = '651_erode.jpg'

img = cv2.imread(os.path.join(path, filename))

img2 = img.copy()

mypoints = [(204, 1449), (200, 2410),(1907, 113), (2154, 114)]

for pt in mypoints:
    cv2.circle(img2, tuple(reversed(pt)), 5, (36,255,12), -1)

cv2.imwrite("C:\\Users\\yzgua\\Desktop\\bat\\erode\\vertice\\651_line_1.jpg", img2)