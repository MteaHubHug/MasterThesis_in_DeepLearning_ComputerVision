import os
import json
import cv2
import numpy as np
import os
import shutil
from os import listdir
from os.path import isfile, join

# 1. copy all files in OK folder
# 2. use JSON file to see classification : OK/ NOK
# those images that are NOK ==> move to NOK folder


def get_classes(input_file):
    json_decode = json.load(input_file)
    classes={}
    for filename in json_decode["_via_img_metadata"]:
        id = json_decode["_via_img_metadata"][filename]["filename"].split("/")[1].split("_")[0].split("-")[0]
        classs = json_decode["_via_img_metadata"][filename]["file_attributes"]["classification"]
        # print(id,classs)
        classes[id]=classs
    return classes


def move_nok(path,new_path,classes):
    files = os.listdir(path)
    cnt=0
    for file in files:
        id=file.split("-")[0]
        if id in classes:
            if(classes[id]=="NOK"):
                print(id,classes[id])
                #original=path + "\\" + file
                #target= new_path + "\\" + file
                #shutil.move(original,target)
                cnt+=1
    return cnt


input_file=open('wuerth_iriis.json', 'r')
path=r'G:\Matea\Base_Line\dataset_temp\usecase Umschlichten\OK'
new_path=r"G:\Matea\Base_Line\dataset_temp\usecase Umschlichten\NOK"

classes=get_classes(input_file)

#for classs in classes:
#   print(classs,classes[classs])

moved=move_nok(path,new_path,classes)
print(moved)

