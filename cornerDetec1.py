import shutil

import cv2

from Configs import SharedConfigurations
import imgaug
conf = SharedConfigurations()

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
import imgaug.augmenters as iaa

from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import os

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5
NUM_KEYPOINTS = 4 * 2  # 4 pairs each having x and y coordinates

IMG_DIR = conf.annotated_IRIIS_images_folder
JSON = conf.IRIIS_json
KEYPOINT_DEF = (
    ""
)

# Load the ground-truth annotations.
json_dict = json.load(JSON)
# Set up a dictionary, mapping all the ground-truth information
# with respect to the path of the image.

def convert_dict(json_dict):
    imdir = r"D:\FINAL DATASET\wuerth_iriis_annotate"
    new_dict={}
    files=os.listdir(imdir)
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

        img_path = IMG_DIR + "\\" + file
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


# Utility for reading an image and for getting its annotations.
def get_box1(name,json_dict,img_dir): # name example : "2101476929-20210908T080118_iriis.jpg"
    filename="wuerth_iriis/" + name + "-1"
    xs=json_dict["_via_img_metadata"][filename]["regions"][0]["shape_attributes"]["all_points_x"]
    ys=json_dict["_via_img_metadata"][filename]["regions"][0]["shape_attributes"]["all_points_y"]
    img_path = img_dir + "\\" + name
    #img_data = plt.imread(img_path)
    img_data= cv2.imread(img_path)
    polyline=[xs,ys]
    #print(polyline)
    return img_data,polyline

def transform_keypoints(img,keypoints):
    p1 = (keypoints[0][0] ,keypoints[1][0])
    p2 = (keypoints[0][1], keypoints[1][1])
    p3 = (keypoints[0][2], keypoints[1][2])
    p4 = (keypoints[0][3], keypoints[1][3])

    keypoints= [p1,p2,p3,p4]
    return keypoints

def visualize_few_examples(samples,json_dict):
    red = [0, 0, 255]
    yellow = [0, 255, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]

    pink= [255,0,255]
    thickness=20
    radius=7

    images=samples
    for im in images:
        fname=im  #### fname example : "2101476929-20210908T080118_iriis.jpg"
        img, keypoints= get_box1(fname,json_dict,IMG_DIR)
        new_keypoints=  transform_keypoints(img,keypoints)

        p1=new_keypoints[0]
        p2=new_keypoints[1]
        p3=new_keypoints[2]
        p4=new_keypoints[3]

        img = cv2.circle(img, p1, radius, pink, thickness)
        img = cv2.circle(img, p2, radius, pink, thickness)
        img = cv2.circle(img, p3, radius, pink, thickness)
        img = cv2.circle(img, p4, radius, pink, thickness)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_samples(img_dir):
    samples=os.listdir(img_dir)
    num_samples = 4
    selected_samples = np.random.choice(samples, num_samples, replace=False)
    #print(selected_samples)
    return samples, selected_samples


def visual(samples,new_dict):
    for file in samples:
        img=new_dict[file]["img_data"]
        imgplot = plt.imshow(img)
        plt.show()

json_dict2=convert_dict(json_dict)
samples, selected_samples= get_samples(IMG_DIR)
##########visualize_few_examples(selected_samples,json_dict)
#visual(selected_samples,json_dict2)

def get_box(name):
    data = json_dict2[name]
    return data


#########################################################################################################
############################### PREPARE DATA GENERATOR ##################################################
#########################################################################################################

class KeyPointsDataset(keras.utils.Sequence):
    def __init__(self, image_keys, aug, batch_size=BATCH_SIZE, train=True):
        self.image_keys = image_keys
        self.aug = aug
        self.batch_size = batch_size
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_keys) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_keys))
        if self.train:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        image_keys_temp = [self.image_keys[k] for k in indexes]
        (images, keypoints) = self.__data_generation(image_keys_temp)

        return (images, keypoints)

    def __data_generation(self, image_keys_temp):
        batch_images = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 3), dtype="int")
        batch_keypoints = np.empty(
            (self.batch_size, 1, 1, NUM_KEYPOINTS), dtype="float32"
        )

        for i, key in enumerate(image_keys_temp):
            data = get_box(key)
            current_keypoint = np.array(data["joints"])[:, :2]
            kps = []

            # To apply our data augmentation pipeline, we first need to
            # form Keypoint objects with the original coordinates.
            for j in range(0, len(current_keypoint)):
                kps.append(Keypoint(x=current_keypoint[j][0], y=current_keypoint[j][1]))

            # We then project the original image and its keypoint coordinates.
            current_image = data["img_data"]
            kps_obj = KeypointsOnImage(kps, shape=current_image.shape)

            # Apply the augmentation pipeline.
            (new_image, new_kps_obj) = self.aug(image=current_image, keypoints=kps_obj)
            batch_images[i,] = new_image

            # Parse the coordinates from the new keypoint object.
            kp_temp = []
            for keypoint in new_kps_obj:
                kp_temp.append(np.nan_to_num(keypoint.x))
                kp_temp.append(np.nan_to_num(keypoint.y))

            # More on why this reshaping later.
            batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, 4 * 2)

        # Scale the coordinates to [0, 1] range.
        batch_keypoints = batch_keypoints / IMG_SIZE

        return (batch_images, batch_keypoints)

############################################### augumentation ###########################################
train_aug = iaa.Sequential(
    [
        iaa.Resize(IMG_SIZE, interpolation="linear"),
        iaa.Fliplr(0.3),
        # `Sometimes()` applies a function randomly to the inputs with
        # a given probability (0.3, in this case).
        iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7))),
    ]
)

test_aug = iaa.Sequential([iaa.Resize(IMG_SIZE, interpolation="linear")])


np.random.shuffle(samples)
train_keys, validation_keys = (
    samples[int(len(samples) * 0.15) :],
    samples[: int(len(samples) * 0.15)],
)

train_dataset = KeyPointsDataset(train_keys, train_aug)
validation_dataset = KeyPointsDataset(validation_keys, test_aug, train=False)

print(f"Total batches in training set: {len(train_dataset)}")
print(f"Total batches in validation set: {len(validation_dataset)}")

sample_images, sample_keypoints = next(iter(train_dataset))
#assert sample_keypoints.max() == 1.0
#assert sample_keypoints.min() == 0.0

sample_keypoints = sample_keypoints[:4].reshape(-1, 4, 2) * IMG_SIZE
#################################################################################
############### model build #####################################################
#################################################################################

def get_model():
    # Load the pre-trained weights of MobileNetV2 and freeze the weights
    backbone = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    backbone.trainable = False

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x)
    x = layers.Dropout(0.3)(x)
    x = layers.SeparableConv2D(
        NUM_KEYPOINTS, kernel_size=5, strides=1, activation="relu"
    )(x)
    outputs = layers.SeparableConv2D(
        NUM_KEYPOINTS, kernel_size=3, strides=1, activation="sigmoid"
    )(x)

    return keras.Model(inputs, outputs, name="keypoint_detector")


get_model().summary()


model = get_model()
model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-4))
model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

################################################################################
#################################### predictions : #############################
################################################################################
#sample_val_images, sample_val_keypoints = next(iter(validation_dataset))
#sample_val_images = sample_val_images[:4]
#sample_val_keypoints = sample_val_keypoints[:4].reshape(-1, 24, 2) * IMG_SIZE
#predictions = model.predict(sample_val_images).reshape(-1, 24, 2) * IMG_SIZE 
