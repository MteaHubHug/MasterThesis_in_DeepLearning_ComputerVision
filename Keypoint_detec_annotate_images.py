from PIL import Image
import tensorflow as tf
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
import json
from PIL import Image
import cv2
import os
from Configs import SharedConfigurations
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt
import numpy as np
import keras

conf = SharedConfigurations()

input_file= open('Keypoints_results.json', 'r')
json_dict=json.load(input_file)
IMG_SIZE = conf.keypoint_detec_IMG_SIZE
IMG_DIR=conf.not_annotated_all_data
RESULTS_DIR=conf.keypoint_detec_results_test_data

def cast_points(str_list):
    new_list=[]
    for item in str_list:
        new_list.append(int(float(item)))
    return new_list

def crop_images(img_dir,json_dict,results_dir):
    images=os.listdir(img_dir)
    for image in images:
        filename= img_dir + "\\" + image
        imname= results_dir + "\\"  +image
        xs=json_dict[image]["all_points_x"]
        ys = json_dict[image]["all_points_y"]
        xs=cast_points(xs)
        ys=cast_points(ys)
        #print(image)
        #print(xs,ys)
        #print("******************")
        the_image = Image.open(filename)
        the_image = the_image.resize((IMG_SIZE, IMG_SIZE))
        top_left_x = min(xs)
        top_left_y = min(ys)
        bot_right_x = max(xs)
        bot_right_y = max(ys)
        box = (top_left_x, top_left_y, bot_right_x, bot_right_y)
        img2 = the_image.crop(box)
        img2 = img2.resize((IMG_SIZE, IMG_SIZE))  # comment this line
        img2.save(imname)


crop_images(IMG_DIR,json_dict,RESULTS_DIR)
