import json
import math
import cv2
import numpy as np

def get_classes(input_file):
    json_decode2 = json.load(input_file)
    classes={}
    for filename in json_decode2["_via_img_metadata"]:
        id = json_decode2["_via_img_metadata"][filename]["filename"].split("/")[1].split("_")[0]
        classs = json_decode2["_via_img_metadata"][filename]["file_attributes"]["classification"]
        # print(id,classs)
        classes[id]=classs
    return classes


def change_json_file(input_file, classes):
    json_decode = json.load(input_file)
    cnt = 0
    for filename in json_decode["_via_img_metadata"]:
        id = json_decode["_via_img_metadata"][filename]["filename"].split("/")[1].split("-")[0]
        classs = json_decode["_via_img_metadata"][filename]["file_attributes"]["classification"]
        if id in classes:
            if (classs != classes[id]):
                cnt += 1
                print(id, classs, classes[id])
                json_decode["_via_img_metadata"][filename]["file_attributes"]["classification"] = classes[id]

    input_file.seek(0)
    json.dump(json_decode, input_file, indent=4)
    input_file.truncate()
    return cnt


input_file2=open('IRIS Nachbesserung 708.json', 'r')
input_file=open('wuerth_iriis.json', 'r+')

#classes=get_classes(input_file2)
#changes=change_json_file(input_file,classes)
#print(changes) #271


def count_2s(input_file):
    json_decode = json.load(input_file)
    cnt=0
    for filename in json_decode["_via_img_metadata"]:
        id = json_decode["_via_img_metadata"][filename]["filename"].split("/")[1].split("_")[0].split("-")[0]
        first_digit=id[0]
        if(first_digit=='2'):
          print(id,first_digit)
          cnt+=1
    return cnt

res1=count_2s(input_file)
res2=count_2s(input_file2)
print(res1,res2)
