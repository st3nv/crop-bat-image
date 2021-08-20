import cv2
import os
import numpy as np
import math

path = r'C:\Users\yzgua\Desktop\bat\erode'
filename = '720_erode.jpg'

img = cv2.imread(os.path.join(path, filename))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
bi = cv2.bilateralFilter(gray, 5, 75, 75)
cv2.namedWindow('bi', 0)
cv2.imshow('bi',bi)

dst = cv2.cornerHarris(bi, 2, 3, 0.04)
#--- create a black image to see where those corners occur ---
mask = np.zeros_like(gray)

#--- applying a threshold and turning those pixels above the threshold to white ---           
mask[dst>0.01*dst.max()] = 255
cv2.namedWindow('mask', 0)
cv2.imshow('mask', mask)


img[dst > 0.01 * dst.max()] = [0, 0, 255]   #--- [0, 0, 255] --> Red ---
cv2.namedWindow('dst', 0)
cv2.imshow('dst', img)

coor = np.argwhere(mask)
coor_list = [l.tolist() for l in list(coor)]
coor_tuples = [tuple(l) for l in coor_list]

def distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

distance_list = [distance((0,0), pct1) for pct1 in coor_tuples]

print(coor_tuples)
print()
print(len(coor_tuples))
print()
print(distance_list)
print(len(distance_list))

pct_num = 200

sorted_dis = sorted(range(len(distance_list)), key=lambda i: distance_list[i])[:pct_num]

print(sorted_dis)

print()

close_points = [coor_tuples[i] for i in sorted_dis]

print(close_points)

thresh = 400

close_points_copy = close_points

i = 1 
for pt1 in close_points:
    # print(' I :', i)
    for pt2 in close_points[i::1]:
        # print(pt1, pt2)
        # print('Distance :', distance(pt1, pt2))
        if(distance(pt1, pt2) < thresh):
            close_points_copy.remove(pt2)      
    i+=1

print(close_points_copy)

img2 = img.copy()

# mypoints = [(90, 202), (2778, 184), (2761, 2890), (107, 2890)]

for pt in close_points_copy:
    cv2.circle(img2, tuple(reversed(pt)), 5, (36,255,12), -1)

# for pt in mypoints:
#      cv2.circle(img2, tuple(pt), 5, (36,255,12), -1)

cv2.imwrite("C:\\Users\\yzgua\\Desktop\\bat\\erode\\vertice\\11.jpg", img2)
cv2.namedWindow('Image with corners', 0)
cv2.imshow('Image with corners', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()