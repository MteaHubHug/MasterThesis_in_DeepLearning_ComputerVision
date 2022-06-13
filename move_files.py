import os
import json
import cv2
import numpy as np
import os
import shutil
from os import listdir
from os.path import isfile, join

from Configs import SharedConfigurations
# 1. copy all files in OK folder
# 2. use JSON file to see classification : OK/ NOK
# those images that are NOK ==> move to NOK folder

configs=SharedConfigurations()

def copy_iriis_images(path,new_path):
    images = os.listdir(path)
    cnt=0
    for image in images:
        if(image[-4:]==".jpg"):
           original=path + "\\" + image
           target= new_path + "\\" + image
           shutil.copy(original,target)
           #print(image,cnt)
           if(cnt%1000==0): print(cnt)
           cnt+=1
    return cnt


def get_classes(input_file):
    json_decode = json.load(input_file)
    classes={}
    for filename in json_decode["_via_img_metadata"]:
        id = json_decode["_via_img_metadata"][filename]["filename"].split("/")[1].split("_")[0].split("-")[0]
        classs = json_decode["_via_img_metadata"][filename]["file_attributes"]["classification"]
        # print(id,classs)      ###### IMPORTANT : THERE ARE NOW OTHER CLASSES BESIDE NOK/OK ---> see if you need to delte them ?!?!?!?!?!
        classes[id]=classs
    return classes


def move_nok(path,new_path,classes):
    files = os.listdir(path)
    cnt=0
    for file in files:
        id=file.split("-")[0]
        if id in classes:
            if(classes[id]=="NOK"):
                print(id,classes[id],cnt)
                original=path + "\\" + file
                target= new_path + "\\" + file
                shutil.move(original,target)
                cnt+=1
    return cnt


input_file=configs.input_file
#old_path=r"G:\Matea\FINAL_DATASET\iriis"
#old_path= r"G:\Matea\triplets"

NV_OK=configs.NV_OK
NV_NOK=configs.NV_NOK
UM_OK=configs.UM_OK
UM_NOK=configs.UM_NOK

#copied=copy_iriis_images(old_path,path)
#print(copied)

#classes=get_classes(input_file)

#for classs in classes:
#   print(classs,classes[classs])

#moved=move_nok(NV_OK,NV_NOK,classes)
#print(moved)

def check_classes(path,input_file): #  ~ delete DAMAGED
    json_decode = json.load(input_file)
    classes={}
    for filename in json_decode["_via_img_metadata"]:
        id = json_decode["_via_img_metadata"][filename]["filename"].split("/")[1].split("_")[0]
        classs = json_decode["_via_img_metadata"][filename]["file_attributes"]["classification"]
        #print(id,classs)
        classes[id] = classs

    files=os.listdir(path)
    damaged=[]
    for file in files:
        id=file.split("_")[0]
        if id in classes:
            classa=classes[id]
            if(classa=="damaged"):
                #damaged.append(id)
                print(id, classes[id])
                #file_path=path + "\\" + id + "_iriis.jpg"
                #os.remove(file_path)







#check_classes(UM_OK,input_file)
#check_classes(UM_OK,input_file)


#########18 591  OK ;;;      2 727 NOK     ;;;  21 318 in SUM



############# In wuerth_iriis.json there are ****17632**** IDs starting with 2
############In IRIS Nachbesserung 708.json there are 17658 IDs starting with 2


###user story requirment
#### try to find up to ****17632**** correspondeces in WUERTH_nachverpacken
#### try to find up to 3084 correspondeces in WUERTH_umschlichten

