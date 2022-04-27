import json
import os
import math
import shutil

import cv2
import numpy as np


sirius_path1=r"G:\Matea\FINAL_DATASET\wuerth_sirius"
sirius_path2=r"G:\Matea\umschlihtung_filtered_images"
path_sirius_color=r"D:\FINAL DATASET\wuerth_sirius"
def copy_sirius_color_images(old_path,new_path):
    files = os.listdir(old_path)
    cnt=0
    for file in files:
        exstension = file[-9:]
        if (exstension == "color.png"):
            old_name=old_path+"\\"+file
            new_name=new_path+"\\"+file
            shutil.copy(old_name,new_name)
            cnt+=1
            if(cnt%100==0): print(cnt)
    print(cnt)

#copy_sirius_color_images(sirius_path2,path_sirius_color)

path_krdis=r"G:\Matea\FINAL_DATASET\wuerth_sirius"
input_file=open('IRIIS Dataset.json', 'r+')

wuetrh_sirius_annotate_path=r"D:\FINAL DATASET\wuerth_sirius_annotate"

def get_ids_iriis(input_file):
    json_decode = json.load(input_file)
    ids=[]
    cnt=0
    for filename in json_decode["_via_img_metadata"]:
        regions=json_decode["_via_img_metadata"][filename]["regions"]
        if(len(regions)>0):
            id = json_decode["_via_img_metadata"][filename]["filename"].split("/")[1].split("_")[0]
            if(id[0]=="4"):
                timestamp=id.split("-")[1]
                id=id.split("-")[0][:-1]
                id=id+"-"+timestamp
            #print(id)
            cnt+=1
            ids.append(id)
    print(cnt)
    return ids

def get_ids_sirius(path): ################## 2101408995-20210907T062247_sirius-color.jpg
    sirius_ims=os.listdir(path)
    ids=[]
    cnt=0
    for im in sirius_ims:
        exstension=im[-9:]
        if(exstension=="color.png"):
            id = im.split("_")[0]
            ids.append(id)
            cnt += 1
            #print(id)
    print(cnt)
    return ids


def get_matches(ids_iriis,ids_sirius):
    matches=[]
    cnt=0
    for id in ids_iriis:
        if id in ids_sirius:
            #print(id)
            matches.append(id)
            cnt+=1
    print(cnt)
    return matches

def copy_images_for_anotations(sirius_path,sirius_annotations_path,matches):
    ims=os.listdir(sirius_path)
    cnt=0
    for im in ims:
        id=im.split("_")[0]
        print(id)
        if id in matches:
            old_name=sirius_path + "\\" + im
            new_name=sirius_annotations_path + "\\" + im
            shutil.copy(old_name,new_name)
            cnt+=1
            print(cnt)
    print(cnt)



ids_iriis=get_ids_iriis(input_file) #1401 annotated iriis images!
ids_sirius=get_ids_sirius(path_sirius_color) ## 16656 sirius color images

matches=get_matches(ids_iriis,ids_sirius) #1132

copy_images_for_anotations(path_sirius_color,wuetrh_sirius_annotate_path,matches)
