import cv2

# Load the image
img = cv2.imread("C://Users//yzgua//Desktop//bat//test_large//720.jpg")
h_pic, w_pic, c_pic = img.shape[:3]
# print(h_pic)

if h_pic ==  3288 and w_pic == 4608:
    cut_h = 100
    cut_w = 850
    img = img[cut_h:h_pic-cut_h, cut_w:w_pic-cut_w]

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 100, 500, L2gradient = True)
# edged = cv2.Canny(img, 170, 490)
cv2.namedWindow('edge', 0)
cv2.imshow('edge',edged)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(edged, 255, 0, 1, 11, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
cv2.namedWindow('thresh', 0)
cv2.imshow('thresh',thresh)

# apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
thresh = cv2.dilate(thresh,None,iterations = 50)
cv2.namedWindow('dilate', 0)
cv2.imshow('dilate',thresh)

thresh = cv2.erode(thresh,None,iterations = 50)
cv2.namedWindow('erode', 0)
cv2.imshow('erode',thresh)


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
#     cv2.rectangle(img,
#             (x,y),(x+w,y+h),
#             (0,0,255),
#             10)
    if( w > maxw and h > maxh and w != maxh and h != maxh):
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

margin_coef = 800

margin_x = int(w_pic/margin_coef)
margin_y = int(h_pic/margin_coef)

ylow = max(baty-margin_y, 0)
yhigh = min(baty+maxh+margin_y, h_pic)
xlow = max(batx-margin_x, 0)
xhigh = min(batx+maxw+margin_x, w_pic)

crop = img[ylow:yhigh, xlow:xhigh]
cv2.imwrite("C:\\Users\\yzgua\\Desktop\\bat\\test\\720\\720_800.jpg", crop)

cv2.namedWindow('crop', 0)
cv2.imshow('crop',crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

