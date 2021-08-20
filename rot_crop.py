# input: line1:[(a1, b1), (c1, d1)]
#       line2:[(a2, b2), (c2, d2)]

import cv2
import numpy as np
import os
import math

def crop_minAreaRect(img, angle, box, h, w):
    # rotate img
    # print(angle)
    rows,cols = img.shape[0], img.shape[1]
    adjust_angle = 0
    if abs(angle-90)<angle:
        adjust_angle = angle-90
    else:
        adjust_angle = angle
    M = cv2.getRotationMatrix2D((cols/2,rows/2),adjust_angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # rotate bounding box
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    print("After rot pts: ", pts)
    pts[pts < 0] = 0
    for i in range(4):
        if pts[i,0] > w:
            pts[i,0] = w
        if pts[i,1] > h:
            pts[i,1] = h
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
    img_crop = img_rot[ylow:yhigh, xlow:xhigh]
    return img_crop



def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)


def my_rot_crop(img, line1, line2):
    h ,w = img.shape[0], img.shape[1]
    squared = 2688

    line1 = list(line1)
    line2 = list(line2)
    
    line1 = [tuple(reversed(pt)) for pt in line1]
    line2 = [tuple(reversed(pt)) for pt in line2]

    print(line1)
    print(line2)

    # Find the realatively vertical line:
    if(abs(line1[0][0] - line1[1][0]) < abs(line2[0][0] - line2[1][0])):
        linev = line1
        lineh = line2
    elif(abs(line1[0][0] - line1[1][0]) > abs(line2[0][0] - line2[1][0])):
        linev = line2
        lineh = line1
    else:
        print("Warning! Cannot distinguish between the lines!")

    print("Linev: ", linev)
    print("Lineh: ", lineh)
    intersec = np.int_(line_intersection(linev, lineh)).tolist()
    print(intersec)
    slopeh = (lineh[0][1] - lineh[1][1]) / (lineh[0][0] - lineh[1][0])
    print("slope: ", slopeh)
    angle_tmp = 90 + math.degrees(math.atan(slopeh))
    print("angle of horizontal line: ", angle_tmp)
    long_edge = np.sin(math.radians(angle_tmp))*squared
    short_edge = np.cos(math.radians(angle_tmp))*squared
    print(long_edge)
    print(short_edge)
    pt1 = intersec
    pt2 = [intersec[0] + long_edge, intersec[1]-short_edge]
    pt3 = [pt2[0]-short_edge, intersec[1]+long_edge]
    pt4 = [intersec[0]+short_edge, intersec[1]+long_edge]
    box_float = np.vstack((pt1,pt2,pt3,pt4))
    box_int = np.int32(box_float)
    print(box_int)
    img_crop = crop_minAreaRect(img, angle_tmp, box_int, h, w)
    cv2.imwrite("C:\\Users\\yzgua\\Desktop\\bat\\erode\\crop\\643.jpg", img_crop)

# test part
path = r'C:\Users\yzgua\Desktop\bat\erode'
filename = '1_erode.jpg'

img = cv2.imread(os.path.join(path, filename))

img2 = img.copy()

line1 = ((130, 152), (129, 188))
line2 = ((171, 104), (206, 105))

my_rot_crop(img2, line1, line2)

# mypoints = [(197, 962), (191, 1892), (1084, 96), (2172, 103)]

# for pt in mypoints:
#     cv2.circle(img2, tuple(reversed(pt)), 5, (36,255,12), -1)

# cv2.circle(img2, intersec, 5, (36,255,12), -1)

# cv2.imwrite("C:\\Users\\yzgua\\Desktop\\bat\\erode\\vertice\\intersec.jpg", img2)