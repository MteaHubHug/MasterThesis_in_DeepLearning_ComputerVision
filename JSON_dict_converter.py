import shutil
import cv2
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

import numpy as np
import json
import os
from PIL import Image
from matplotlib import pyplot as plt
from Configs import SharedConfigurations
conf = SharedConfigurations()

IMG_DIR = conf.annotated_IRIIS_images_folder
JSON = conf.IRIIS_json
json_dict = json.load(JSON)


def get_box1(name):
    filename = "wuerth_iriis/" + name + "-1"
    data = json_dict[filename]
    img_path = IMG_DIR + "\\" + name
    img_data = plt.imread(img_path)
    # If the image is RGBA convert it to RGB.
    if img_data.shape[-1] == 4:
        img_data = img_data.astype(np.uint8)
        img_data = Image.fromarray(img_data)
        img_data = np.array(img_data.convert("RGB"))
    data["img_data"] = img_data

    return data


def convert_dict(json_dict, img_dir):
    new_dict={}
    files=os.listdir(img_dir)
    for file in files:
        inner_dict={}
        filename="wuerth_iriis/" + file + "-1"
        xs = json_dict["_via_img_metadata"][filename]["regions"][0]["shape_attributes"]["all_points_x"]
        ys = json_dict["_via_img_metadata"][filename]["regions"][0]["shape_attributes"]["all_points_y"]

        p1 = [xs[0],ys[0],1]
        p2 = [xs[1],ys[1],1]
        p3 = [xs[2],ys[2],1]
        p4 = [xs[3],ys[3],1]
        keypoints= [p1,p2,p3,p4]

        img_path = img_dir + "\\" + file
        img_data = plt.imread(img_path)
        # If the image is RGBA convert it to RGB.
        if img_data.shape[-1] == 4:
            img_data = img_data.astype(np.uint8)
            img_data = Image.fromarray(img_data)
            img_data = np.array(img_data.convert("RGB"))

        inner_dict["joints"]=keypoints
        inner_dict["img_data"]=img_data
        new_dict[file]=inner_dict
    return new_dict

new_dict=convert_dict(json_dict,IMG_DIR)

def print_dict(new_dict,img_dir):
    files=os.listdir(img_dir)
    for file in files:
        joints=new_dict[file]["joints"]
        print(joints)

print_dict(new_dict,IMG_DIR)