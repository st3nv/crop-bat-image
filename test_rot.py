import cv2

# Load the image
img = cv2.imread("C://Users//yzgua//Desktop//bat//pic//rot//2_rot.jpg")
h_pic, w_pic, c_pic = img.shape[:3]
print(h_pic)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(img, 170, 490)
# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 11, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
thresh = cv2.dilate(thresh,None,iterations = 15)
thresh = cv2.erode(thresh,None,iterations = 15)

# Find the contours
contours,hierarchy = cv2.findContours(thresh,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it
batx = 0
baty = 0
maxw = 0
maxh = 0

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,
            (x,y),(x+w,y+w),
            (0,0,255),
            10)
    if( w > maxw and h > maxh and w != w_pic):
        batx = x
        baty = y
        maxw = w
        maxh = h

# cv2.rectangle(img,
#             (batx,baty),(batx+maxw,baty+maxh),
#             (0,0,255),
#             10)
    
cv2.namedWindow('demo', 0)
cv2.imshow('demo',img)
cv2.waitKey(0)
cv2.destroyAllWindows()