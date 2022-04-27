import os
import cv2
from cv2 import  perspectiveTransform
from cv2 import transform
import json
import numpy as np
from Configs import SharedConfigurations

configs=SharedConfigurations()

IRIIS_json=configs.IRIIS_json
SIRIUS_json=configs.SIRIUS_json

def get_IRIIS_boxCorners(input_file):
    json_decode = json.load(input_file)
    corners = {}
    for filename in json_decode["_via_img_metadata"]:
        id = json_decode["_via_img_metadata"][filename]["filename"].split("/")[1].split("_")[0]  # id example : 41000103322-20210907T053847  # len(41000103322)=11 :(
        timestamp=id.split("-")[1]
        if(id[0]=="4"):
            id=id.split("-")[0][:-1]+"-"+timestamp  # id examle now : 4100010332-20210907T053847 # len(4100010332)=10 :)
        regions = json_decode["_via_img_metadata"][filename]["regions"]
        if(len(regions)>0):
            x_coords=regions[0]["shape_attributes"]["all_points_x"]
            y_coords=regions[0]["shape_attributes"]["all_points_y"]
            #print(id, " * x: * ", x_coords, " * y: * ", y_coords)
            corners[id]=[x_coords,y_coords]
    #for corner in corners:
    #    print(corner, "*********", corners[corner])
    return  corners

def get_SIRIUS_boxCorners(input_file):
    json_decode = json.load(input_file)
    corners = {}
    for filename in json_decode["_via_img_metadata"]:
        id = json_decode["_via_img_metadata"][filename]["filename"].split("_")[0]   # id_example : 4100009693-20210912T091226 # len(4100009693)=10 :))
        regions = json_decode["_via_img_metadata"][filename]["regions"]
        x_coords = regions[0]["shape_attributes"]["all_points_x"]
        y_coords = regions[0]["shape_attributes"]["all_points_y"]
        corners[id] = [x_coords, y_coords]
    #for corner in corners:
    #    print(corner, "*********", corners[corner])
    return  corners

def get_corresponding_SIRIUSxIRIIScorners(iriis_corners,sirius_corners):
    cnt=0
    siriusXiriisCorners={}
    for id in sirius_corners:
        if id in iriis_corners:
            cnt+=1
            #print(id, "**IRIIS : **" ,iriis_corners[id], "** SIRIUS:**", sirius_corners[id])
            siriusXiriisCorners[id]=[ iriis_corners[id],  sirius_corners[id]  ]
    #for id in siriusXiriisCorners:
    #    print(id, "*********", siriusXiriisCorners[id])
    print(cnt) #1132
    return siriusXiriisCorners

def get_just_few_examples(corners_dict):
    examples = {}
    i=0
    for id in corners_dict:
        examples[id] = corners_dict[id]
        i += 1
        if (i == 5): break
    for id in examples:
        print(id, "*****",examples[id])
    return examples

iriis_corners=get_IRIIS_boxCorners(IRIIS_json)
sirius_corners=get_SIRIUS_boxCorners(SIRIUS_json)

siriusXiriis_corners=get_corresponding_SIRIUSxIRIIScorners(iriis_corners,sirius_corners)

examples=get_just_few_examples(siriusXiriis_corners)

"""
2101408996-20210913T051607 ***** [[[142, 1225, 1189, 101], [53, 69, 751, 740]], [[949, 933, 388, 403], [111, 976, 968, 97]]]
2101416256-20210906T115244 ***** [[[517, 1185, 1167, 507], [201, 204, 718, 747]], [[813, 823, 420, 414], [422, 931, 941, 429]]]
2101416425-20210906T113338 ***** [[[508, 1177, 1145, 483], [201, 218, 732, 750]], [[815, 814, 411, 412], [433, 938, 939, 431]]]
2101416472-20210906T114643 ***** [[[532, 1197, 1178, 522], [205, 204, 715, 753]], [[814, 824, 424, 414], [421, 927, 941, 431]]]
2101416518-20210906T113736 ***** [[[510, 1180, 1168, 507], [203, 194, 713, 754]], [[812, 833, 431, 413], [415, 921, 940, 429]]]
"""
transformation_examples_path=r"D:\transformation_examples"
saved_examples_path=r"D:\saved_examples"

def resize_images_and_save(path, saving_path):
    images=os.listdir(path)
    for im in images:
        image_path=path+"\\"+im
        img = cv2.imread(image_path)
        dimensions=img.shape   # IRIIS DIMS : (800, 1280, 3)  ;; SIRIUS DIMS : (1024, 1360, 3)  --- DIMS ARE ( HEIGHT, WIDTH)
        extension=im[-9:]
        if(extension=="iriis.jpg"):
            width = 1360
            height = 1024
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            dimensions = img.shape
        print(dimensions)
        filename=saving_path+"\\"+im
        cv2.imwrite(filename, img)

resize_images_and_save(transformation_examples_path, saved_examples_path)

# now work with saved images ===> folder : saved_examples_path=r"D:\saved_examples
wd=saved_examples_path

def draw_sirius_corners_on_iriis_image(path,corners_dict):
    pink = [199, 21, 133]
    yellow = [255, 222, 173]
    green = [64, 224, 208]
    blue = [30, 144, 255]

    images=os.listdir(path)
    for im in images:
        extension=im[-9:]
        if(extension=="iriis.jpg"):
            image_path = path + "\\" + im
            id=im.split("_")[0]
            timestamp = id.split("-")[1]
            if (id[0] == "4"):
                id = id.split("-")[0][:-1] + "-" + timestamp  # id examle now : 4100010332-20210907T053847 # len(4100010332)=10 :)

            if id in corners_dict:
                sirius_edges=corners_dict[id][1]
                sirius_x=sirius_edges[0]
                sirius_y=sirius_edges[1]
                print(id, "*******", sirius_edges,sirius_x,sirius_y)
                a=sirius_x[0], sirius_y[0]
                b=sirius_x[1], sirius_y[1]
                c=sirius_x[2], sirius_y[2]
                d=sirius_x[3], sirius_y[3]

            #img = cv2.imread(image_path)
            #img = cv2.circle(img,centerOfCircle , 3, red, 3)


draw_sirius_corners_on_iriis_image(wd,examples)