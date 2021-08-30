import cv2
import numpy as np
import pandas as pd
import json
import math
import itertools
from collections import Counter

def rec_crop(path, filename, rootpath):
    # Load the image
    img = cv2.imread(path)
    h_pic, w_pic, c_pic = img.shape[:3]
    # print(h_pic)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(img, 170, 490)
    # cv2.namedWindow('edge', 0)
    # cv2.imshow('edge',edged)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 0, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    # cv2.namedWindow('thresh', 0)
    # cv2.imshow('thresh',thresh)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations = 15)
    # cv2.namedWindow('dilate', 0)
    # cv2.imshow('dilate',thresh)

    thresh = cv2.erode(thresh,None,iterations = 15)
    # cv2.namedWindow('erode', 0)
    # cv2.imshow('erode',thresh)

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
        if( w > maxw and h > maxh):
            batx = x
            baty = y
            maxw = w
            maxh = h

    # cv2.rectangle(img,
    #             (batx,baty),(batx+maxw,baty+maxh),
    #             (0,0,255),
    #             5)
        
    # cv2.namedWindow('demo', 0)
    # cv2.imshow('demo',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    margin_coef = 10000

    margin_x = int(w_pic/margin_coef)
    margin_y = int(h_pic/margin_coef)

    ylow = max(baty-margin_y, 0)
    yhigh = min(baty+maxh+margin_y, h_pic)
    xlow = max(batx-margin_x, 0)
    xhigh = min(batx+maxw+margin_x, w_pic)

    crop = img[ylow:yhigh, xlow:xhigh]
    out_path = rootpath + "\\crop\\"+ filename +".jpg"
    # print(out_path)
    cv2.imwrite(out_path, crop)
    return [xlow, ylow, xhigh-xlow, yhigh-ylow]

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
    angle = rect[2]
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

def rot_crop_old(path, filename, root_path):
    # Load the image
    img = cv2.imread(path)
    h, w, c = img.shape[:3]
    # print("height: {} , width: {}".format(h, w))

    # cv2.namedWindow('original', 0)
    # cv2.imshow('original',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(img, 100, 600)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations = 50)
    thresh = cv2.erode(thresh,None,iterations = 50)

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
        box_w = box[1,0]- box[0,0]+1
        if tmparea > maxarea:
            maxbox = box
            maxarea = tmparea
            maxrect = rect

    # cv2.drawContours(img,[maxbox],0,(0,0,255),10)
    # cv2.namedWindow('image bound', 0)
    # cv2.imshow('image bound',img)
    crop_rot = crop_minAreaRect(img, maxrect, maxbox, h , w)
    outpath = root_path + "\\result\\"+ filename +".jpg"
    # print(outpath)

    # cv2.namedWindow('Crop result', 0)
    # cv2.imshow('Crop result',crop_rot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("Maxbox:")
    print(maxbox)
    print(maxrect)
    cv2.imwrite(outpath, crop_rot)
    return maxbox[0], maxrect[2], crop_rot.shape[0], crop_rot.shape[1]

def distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

def rcg_vertice(img):
    pct_num = 200
    thresh = 300

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    bi = cv2.bilateralFilter(img, 5, 75, 75)
    # cv2.namedWindow('bi', 0)
    # cv2.imshow('bi',bi)

    dst = cv2.cornerHarris(bi, 2, 3, 0.04)
    #--- create a black image to see where those corners occur ---
    mask = np.zeros_like(img)

    #--- applying a threshold and turning those pixels above the threshold to white ---           
    mask[dst>0.01*dst.max()] = 255
    # cv2.namedWindow('mask', 0)
    # cv2.imshow('mask', mask)

    # img[dst > 0.01 * dst.max()] = [0, 0, 255]   #--- [0, 0, 255] --> Red ---
    # # cv2.namedWindow('dst', 0)
    # # cv2.imshow('dst', img)

    coor = np.argwhere(mask)
    coor_list = [l.tolist() for l in list(coor)]
    coor_tuples = [tuple(l) for l in coor_list]
    distance_list = [distance((0,0), pct1) for pct1 in coor_tuples]
    sorted_dis = sorted(range(len(distance_list)), key=lambda i: distance_list[i])[:pct_num]
    close_points = [coor_tuples[i] for i in sorted_dis]

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
    return close_points_copy

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return min(360 - ang_deg, 180-(360-ang_deg))
    else: 

        return min(ang_deg, 180-ang_deg)

def rcg_angle(pntlist):
    line1_maxang = ()
    line2_maxang = ()
    angmax = -1
    for pnt4 in itertools.combinations(pntlist, r=4):
        # print("All 4 comb: ", pnt4)
        for line1 in itertools.combinations(pnt4, r=2):
            cpart = Counter(line1)
            call = Counter(pnt4)
            line2 = tuple(list((call-cpart).elements()))
            # print("Line 1: ", line1)
            # print("Line 2: ", line2)
            angtmp = ang(line1, line2)
            if angtmp > angmax:
                angmax = angtmp
                line1_maxang = line1
                line2_maxang = line2
            # print("Angle between: ", angtmp)
            # print("\n")

    # print("Max angle: ", angmax)
    # print("Line 1: ", line1_maxang)
    # print("Line 2: ", line2_maxang)
    return (line1_maxang, line2_maxang)

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

def crop_anglebox(img, angle, box, h, w):
    # rotate img
    # print(angle)
    print(box)
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

def linecrop(img, line1, line2):
    h ,w = img.shape[0], img.shape[1]
    squared = 2675

    line1 = list(line1)
    line2 = list(line2)
    
    line1 = [tuple(reversed(pt)) for pt in line1]
    line2 = [tuple(reversed(pt)) for pt in line2]

    # Find the realatively vertical line:
    if(abs(line1[0][0] - line1[1][0]) < abs(line2[0][0] - line2[1][0])):
        linev = line1
        lineh = line2
    elif(abs(line1[0][0] - line1[1][0]) > abs(line2[0][0] - line2[1][0])):
        linev = line2
        lineh = line1
    else:
        print("Warning! Cannot distinguish between the lines!")

    # print("Linev: ", linev)
    # print("Lineh: ", lineh)
    intersec = np.int_(line_intersection(linev, lineh)).tolist()
    # print("intersection: ", intersec)
    slopeh = (lineh[0][1] - lineh[1][1]) / (lineh[0][0] - lineh[1][0])
    # print("slope: ", slopeh)
    angle_tmp = 90 + math.degrees(math.atan(slopeh))
    # print("angle of horizontal line: ", angle_tmp)
    long_edge = np.sin(math.radians(angle_tmp))*squared
    short_edge = np.cos(math.radians(angle_tmp))*squared
    # print(long_edge)
    # print(short_edge)
    pt1 = intersec
    pt2 = [intersec[0] + long_edge, intersec[1]-short_edge]
    pt3 = [pt2[0] + short_edge, pt2[1]+long_edge]
    pt4 = [intersec[0]+short_edge, intersec[1]+long_edge]
    box_float = np.vstack((pt1,pt2,pt3,pt4))
    box_int = np.int32(box_float)
    img_crop = crop_anglebox(img, angle_tmp, box_int, h, w)
    return img_crop, intersec, angle_tmp

def rot_crop(img, filename, root_path):
    img_crop = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(img, 100, 600)
    # cv2.namedWindow('edge', 0)
    # cv2.imshow('edge',edged)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    # cv2.namedWindow('thresh', 0)
    # cv2.imshow('thresh',thresh)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations = 50)
    # cv2.namedWindow('dilate', 0)
    # cv2.imshow('dilate',thresh)

    thresh = cv2.erode(thresh,None,iterations = 50)

    outpath_erode = root_path + "\\crop_erode\\"+ filename +".jpg"

    erode = thresh.copy()
    cv2.imwrite(outpath_erode, erode)

    # cv2.namedWindow('erode', 0)
    # cv2.imshow('erode',thresh)
    vertice_list = rcg_vertice(erode)
    line1, line2 = rcg_angle(vertice_list)
    img_rot_crop, intersec, angle = linecrop(img_crop, line1, line2)

    outpath_final = root_path + "\\result\\"+ filename +".jpg"
    cv2.imwrite(outpath_final, img_rot_crop)

    return intersec, angle, img_rot_crop.shape[0], img_rot_crop.shape[1]

def label_change(anno, filename, images_id, x_cut, y_cut, intersection, angle, img_aftercrop_h, img_aftercrop_w, json_data):    
    if type(anno.loc[images_id.loc[filename][2]].bbox).__name__=='list':
        images_bbox = anno.loc[images_id.loc[filename][2]].bbox
    else:
        images_bbox=anno.loc[images_id.loc[filename][2]].bbox.values

    # 查找filename的源images_bbox
    ori_label = images_bbox

    rec_cut = [x_cut, y_cut]

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

    # 变换后的的bbox
    output_bbox = [xlow, ylow, xhigh-xlow, yhigh-ylow]
    # print(output_bbox)

    # P.S. Here need to write the output_bbox to the output dataframe.
    ...change the json_data

def rot_crop_all(path, filename, root_path):
    # Load the image
    img_origin = cv2.imread(path)
    # cv2.namedWindow('original', 0)
    # cv2.imshow('original',img_origin)

    img = img_origin.copy()

    h, w, c = img.shape[:3]
    print("height: {} , width: {}".format(h, w))

    cut_w = 0
    cut_h = 0
    intersec = [0,0]
    angle = 90
    after_crop_w = 0
    after_crop_h = 0

    # For spesific pics use different methods
    if h ==  3288 and w == 4608:
        cut_h = 100
        cut_w = 850
        img = img[cut_h:h-cut_h, cut_w:w-cut_w]
        intersec, angle, after_crop_h, after_crop_w = rot_crop(img, filename, root_path)
    else:
        box_rec = rec_crop(path,filename, root_path)
        cut_w = box_rec[0]
        cut_h = box_rec[1]
        path_rec = root_path +"\\crop\\"+ filename +".jpg"
        intersec, angle, after_crop_h, after_crop_w = rot_crop_old(path_rec,filename, root_path)
    
    # print("----------------------------------------")
    # print("cut_w: ", cut_w)
    # print("cut_h: ", cut_h)
    # print("intersec: ", intersec)
    # print("angle: ", angle)
    # print("after_crop_h: ", after_crop_h)
    # print("after_crop_w: ", after_crop_w)
    # print("----------------------------------------")

    # Change label
    # change path
    with open('C:/Users/yzgua/Desktop/bat/train.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
        images_id=pd.DataFrame(json_data['images'])
        images_id.index=images_id.file_name
        anno=pd.DataFrame(json_data['annotations'])
        anno.index=anno.image_id

    # use label_change function to Modify the json_data output
    label_change(anno, filename, images_id, cut_w, cut_h, intersec, angle, after_crop_h, after_crop_w, json_data)

    # 以下为 trimming_zangwu_json中原来的代码
    # # 保存至json文件
    # class NumpyEncoder(json.JSONEncoder):
    #     """ Special json encoder for numpy types """
    #     def default(self, obj):
    #         if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
    #                             np.int16, np.int32, np.int64, np.uint8,
    #                             np.uint16, np.uint32, np.uint64)):
    #             return int(obj)
    #         elif isinstance(obj, (np.float_, np.float16, np.float32,
    #                             np.float64)):
    #             return float(obj)
    #         elif isinstance(obj, (np.ndarray,)):
    #             return obj.tolist()
    #         return json.JSONEncoder.default(self, obj)

    # json_data['annotations']=list(anno.reset_index(drop=True).T.to_dict().values())
    # with open(tar+'/train.json',"w") as f:
    #     json.dump(json_data,f, cls=NumpyEncoder)
    #     print("加载入文件完成...")






