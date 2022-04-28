import os
import cv2
from cv2 import perspectiveTransform
from cv2 import transform
import json
import numpy as np
from math import atan2, degrees
from Configs import SharedConfigurations
from PIL import Image, ImageDraw
configs=SharedConfigurations()

IRIIS_json=configs.IRIIS_json
SIRIUS_json=configs.SIRIUS_json

red = [0, 0, 255]
yellow = [0, 255, 255]
green = [0, 255, 0]
blue = [255, 0, 0]
pink = [255, 0, 255]
pink = [255, 0, 255]
thickness = 20
radius = 7

newsize = (1280, 800)





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


def GetAngleOfLineBetweenTwoPoints(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))

def copy_and_rotate_iriis_images(path,saving_path,corners_dict):
    images=os.listdir(path)

    for im in images:  # IRIIS DIMS : (800, 1280, 3)  ;; SIRIUS DIMS : (1024, 1360, 3)  --- DIMS ARE ( HEIGHT, WIDTH)
        extension = im[-9:]
        if (extension == "iriis.jpg"):
            id=im.split("_")[0]
            timestamp = id.split("-")[1]
            if (id[0] == "4"):
                id = id.split("-")[0][:-1] + "-" + timestamp
            if id in corners_dict:
                image_path = path + "\\" + im
                img = cv2.imread(image_path)


                iriis_edges=corners_dict[id][0]
                iriis_x=iriis_edges[0]
                iriis_y=iriis_edges[1]
                #print(id, "*******", sirius_edges,sirius_x,sirius_y)
                a= ( iriis_x[0], iriis_y[0] )
                b= ( iriis_x[1], iriis_y[1] )
                c= ( iriis_x[2], iriis_y[2] )
                d= ( iriis_x[3], iriis_y[3] )

                #img = cv2.circle(img, a , radius, red, thickness)
                #img = cv2.circle(img, b, radius,  yellow, thickness)
                #img = cv2.circle(img, c, radius,  green, thickness)
                #img = cv2.circle(img, d, radius,  blue, thickness)

                angle=GetAngleOfLineBetweenTwoPoints(a,b)

                filename = saving_path + "\\" + im
                cv2.imwrite(filename, img)

                img2 = Image.open(filename)
                img2 = img2.rotate(angle)
                #filename2 = saving_path + "\\" + "_rot_" + im
                filename2 = saving_path + "\\"  + im
                img2.save(filename2)


def copy_sirius_images(path,saving_path,corners_dict):
    images=os.listdir(path)
    for im in images:
            id=im.split("_")[0]

            if id in corners_dict:
                image_path = path + "\\" + im
                img = cv2.imread(image_path)
                filename = saving_path + "\\" + im
                cv2.imwrite(filename, img)


def crop_and_save(path,corners_dict):
    images=os.listdir(path)
    for im in images:
        id = im.split("_")[0]
        extension = im[-9:]
        if (extension == "iriis.jpg"):
            timestamp = id.split("-")[1]
            if (id[0] == "4"):
                id = id.split("-")[0][:-1] + "-" + timestamp

            if id in corners_dict:
                    image_path = path + "\\" + im
                    img = Image.open(image_path)

                    iriis_edges = corners_dict[id][0]
                    iriis_x = iriis_edges[0]
                    iriis_y = iriis_edges[1]
                    # print(id, "*******", sirius_edges,sirius_x,sirius_y)
                    a = (iriis_x[0], iriis_y[0])
                    b = (iriis_x[1], iriis_y[1])
                    c = (iriis_x[2], iriis_y[2])
                    d = (iriis_x[3], iriis_y[3])

                    img = img.crop((a[0], a[1], c[0], c[1]))
                    #filename = path + "\\" + "_crop_" + im
                    filename = path + "\\"  + im
                    img.save(filename)
        elif(extension=="color.png"):
            if id in corners_dict:
                image_path = path + "\\" + im
                img = cv2.imread(image_path)

                sirius_edges=corners_dict[id][1]
                sirius_x=sirius_edges[0]
                sirius_y=sirius_edges[1]
                p1= ( sirius_x[0], sirius_y[0] )
                p2= ( sirius_x[1], sirius_y[1] )
                p3= ( sirius_x[2], sirius_y[2] )
                p4= ( sirius_x[3], sirius_y[3] )

                #img = cv2.circle(img, p1, radius, pink, thickness)
                #img = cv2.circle(img, p2, radius, pink, thickness)
                #img = cv2.circle(img, p3, radius, pink, thickness)
                #img = cv2.circle(img, p4, radius, pink, thickness)

                filename = path + "\\" + im
                cv2.imwrite(filename, img)
                img = Image.open(image_path)
                img = img.crop((p4[0], p4[1], p2[0], p2[1]))


                # filename = path + "\\" + "_crop_" + im
                filename = path + "\\" + im
                img.save(filename)

                #### roatate SIRIUS image :
                img = cv2.imread(image_path)
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(filename, img)


def resize_and_save_images(path,corners_dict):
    images=os.listdir(path)
    for im in images:
        id = im.split("_")[0]
        extension = im[-9:]
        if (extension == "iriis.jpg"):
            timestamp = id.split("-")[1]
            if (id[0] == "4"):
                id = id.split("-")[0][:-1] + "-" + timestamp
        if id in corners_dict:
            image_path = path + "\\" + im
            img = Image.open(image_path)


            img = img.resize(newsize)

            filename = path + "\\" + im
            img.save(filename)


##################################################################################
iriis_corners=get_IRIIS_boxCorners(IRIIS_json)
sirius_corners=get_SIRIUS_boxCorners(SIRIUS_json)

siriusXiriis_corners=get_corresponding_SIRIUSxIRIIScorners(iriis_corners,sirius_corners)

#examples=get_just_few_examples(siriusXiriis_corners)

iriis_path=r"D:\FINAL DATASET\wuerth_iriis"
sirius_path=r"D:\FINAL DATASET\wuerth_sirius"
transformation_examples_path=r"D:\transformation_examples"
saved_examples_path=r"D:\saved_examples"


#copy_and_rotate_iriis_images(iriis_path,saved_examples_path,siriusXiriis_corners)
#copy_sirius_images(sirius_path,saved_examples_path,siriusXiriis_corners)
#crop_and_save(saved_examples_path,siriusXiriis_corners)


resize_and_save_images(saved_examples_path,siriusXiriis_corners)