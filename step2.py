import cv2
import numpy as np

def calarea(rec):
    x1 = rec[0,0]
    y1 = rec[0,1]
    x2 = rec[1,0]
    y2 = rec[1,1]
    x3 = rec[2,0]
    y3 = rec[2,1]
    x4 = rec[3,0]
    y4 = rec[3,1]
    area = ((x1*y2-y1*x2)+(x2*y3-y2*x3)+(x3*y4-y3*x4)+(x4*y1-y4*x1))/2
    return abs(area)

def crop_minAreaRect(img, rect, box, h, w):
    # rotate img
    print("rect:")
    print(rect)
    print(type(box))
    print("box:")
    print(box)
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle-90,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # rotate bounding box
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    print("After rot pts: ", pts)
    pts[pts < 0] = 0
    for i in range(4):
        if pts[i,0] > w:
            pts[i,0] = w
        if pts[i,1] >h:
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

# Load the image
img = cv2.imread("C://Users//yzgua//Desktop//bat//pic//crop//1_crop.jpg")
h, w, c = img.shape[:3]
print("height: {} , width: {}".format(h, w))

cv2.namedWindow('original', 0)
cv2.imshow('original',img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(img, 170, 500)
cv2.namedWindow('edge', 0)
cv2.imshow('edge',edged)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 11, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

cv2.namedWindow('thresh', 0)
cv2.imshow('thresh',thresh)

# apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
thresh = cv2.dilate(thresh,None,iterations = 10)
cv2.namedWindow('dilate', 0)
cv2.imshow('dilate',thresh)

thresh = cv2.erode(thresh,None,iterations = 10)
cv2.namedWindow('erode', 0)
cv2.imshow('erode',thresh)

# Find the contours
contours,hierarchy = cv2.findContours(thresh,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

maxarea = 0
maxbox = np.zeros((4,2), int)
maxrect = cv2.minAreaRect(contours[0])

for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box_float = cv2.boxPoints(rect)
    tmparea = calarea(box_float)
    box = np.int0(box_float)
    cv2.drawContours(img,[box],0,(0,0,255),10)
    box_w = box[1,0]- box[0,0]+1
    if tmparea > maxarea:
        maxbox = box
        maxarea = tmparea
        maxrect = rect

# cv2.drawContours(img,[maxbox],0,(0,0,255),10)
cv2.namedWindow('image bound', 0)
cv2.imshow('image bound',img)

crop_rot = crop_minAreaRect(img, maxrect, maxbox, h , w)
print(type(crop_rot))
print(crop_rot.shape)
cv2.namedWindow('Crop result', 0)
cv2.imshow('Crop result',crop_rot)
# cv2.imwrite("C:\\Users\\yzgua\\Desktop\\bat\\test\\720\\result\\720_800_result.jpg", crop_rot)
cv2.waitKey(0)
cv2.destroyAllWindows()