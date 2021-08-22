import numpy as np
import cv2
import os

img_aftercrop_w = 2675
img_aftercrop_h = 2675

# label alternation
x_lab = 2000
y_lab = 1500
w_lab = 400
h_lab = 500

ori_label = [x_lab, y_lab, w_lab, h_lab]

x_cut = 850
y_cut = 100
w_cut = 2908
h_cut = 3088

rec_cut = [x_cut, y_cut, w_cut, h_cut]

intersection = [115, 141]
angle = 88.87301527452207


ori_label[0] = ori_label[0] - rec_cut[0] - intersection[0]
ori_label[1] = ori_label[1] - rec_cut[1] - intersection[1]

if abs(angle-90)<angle:
    adjust_angle = angle-90
else:
    adjust_angle = angle

box = [[ ori_label[0], ori_label[1]],
        [ori_label[0]+ori_label[2], ori_label[1]],
        [ori_label[0]+ori_label[2], ori_label[1]+ori_label[3]],
        [ori_label[0], ori_label[1]+ori_label[3]]]

M = cv2.getRotationMatrix2D((0,0),adjust_angle,1)
# rotate bounding box
pts = np.int0(cv2.transform(np.array([box]), M))[0]

print(pts)

pts[pts < 0] = 0

for i in range(4):
    if pts[i,0] > img_aftercrop_w:
        pts[i,0] = img_aftercrop_w
    if pts[i,1] > img_aftercrop_h:
        pts[i,1] = img_aftercrop_h

x1 = pts[0,0]
y1 = pts[0,1]
x2 = pts[1,0]
y2 = pts[1,1]
x3 = pts[2,0]
y3 = pts[2,1]
x4 = pts[3,0]
y4 = pts[3,1]

xlow = min(x1, x2, x3, x4)
xhigh = max(x1, x2, x3, x4)
ylow = min(y1, y2, y3, y4)
yhigh = max(y1, y2, y3, y4)

alt_label = [xlow, ylow, xhigh-xlow, yhigh-ylow]

print(alt_label)