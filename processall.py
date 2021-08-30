import os
from functions import *

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
    # Location of the pic folder
    root_path = "C:\\Users\\yzgua\\Desktop\\bat\\final_test"
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
    rot_crop_all(path,filename, root_path)