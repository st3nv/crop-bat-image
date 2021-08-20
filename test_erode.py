import cv2
import os

def erode(path, filename, root_path):
    # Load the image
    img_origin = cv2.imread(path)
    cv2.namedWindow('original', 0)
    cv2.imshow('original',img_origin)

    img = img_origin.copy()
    # img_crop = img_origin.copy()

    h, w, c = img.shape[:3]
    # print("height: {} , width: {}".format(h, w))

    # if h ==  3288 and w == 4608:
    #     cut_h = 100
    #     cut_w = 850
    #     img = img[cut_h:h-cut_h, cut_w:w-cut_w]
    #     img_crop = img_crop[cut_h:h-cut_h, cut_w:w-cut_w]


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(img, 100, 300)
    cv2.namedWindow('edge', 0)
    cv2.imshow('edge',edged)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow('thresh', 0)
    cv2.imshow('thresh',thresh)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations = 25)
    cv2.namedWindow('dilate', 0)
    cv2.imshow('dilate',thresh)

    thresh = cv2.erode(thresh,None,iterations = 25)

    outpath_erode = root_path + "\\erode\\"+ filename +".jpg"

    erode = thresh.copy()
    cv2.imwrite(outpath_erode, erode)

def get_file_path(root_path,file_list,dir_list,fnlist):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # get_file_path(dir_file_path,file_list,dir_list)
        else:
            fnlist.append(dir_file)
            file_list.append(dir_file_path)
 
if __name__ == "__main__":
    root_path = "C:\\Users\\yzgua\\Desktop\\bat\\test_erode"
    file_list = []
    dir_list = []
    fnlist = []
    get_file_path(root_path,file_list,dir_list, fnlist)
    

num = len(file_list)

## Old code

# for i in range(num):
#     path = file_list[i]
#     filename = fnlist[i][:-4]
#     print("path:", path)
#     rec_crop(path,filename, root_path)
#     path_rec = root_path +"\\crop\\"+ filename +".jpg"
#     print(path_rec)
#     rot_crop(path_rec,filename, root_path)


## New code

for i in range(num):
    path = file_list[i]
    filename = fnlist[i][:-4]
    print("path:", path)
    erode(path,filename, root_path)