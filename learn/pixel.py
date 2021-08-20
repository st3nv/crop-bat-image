import argparse
import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("-i","--image",required=True,
#               help = "path to input image")
# args = vars(ap.parse_args)

image = cv2.imread(r"C:\Users\yzgua\Desktop\bat\pic\1.jpg")

cv2.namedWindow('demo', 0)
cv2.imshow("demo", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

(h, w) = image.shape[:2]
print("height: {}".format(h))

(b, g, r) = image[0,0]
print("bgr of the image: {}{}{}".format(b,g,r))


(cX, cY) = (w//2, h//2)

tr = image[0:cY, cX:w]


cv2.imshow("tr", tr)
cv2.resizeWindow('tr', 600, 500)
cv2.waitKey(0)
cv2.destroyAllWindows()