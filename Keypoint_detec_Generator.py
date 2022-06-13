from Configs import SharedConfigurations
from tensorflow import keras
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import json
import os

conf = SharedConfigurations()
IMG_SIZE = conf.keypoint_detec_IMG_SIZE
BATCH_SIZE = conf.keypoint_detec_BATCH_SIZE
NUM_KEYPOINTS = conf.num_keypoints
IMG_DIR = conf.annotated_IRIIS_images_folder
JSON = conf.IRIIS_json
json_dict = json.load(JSON)


def get_samples(img_dir):
    samples = os.listdir(img_dir)
    num_samples = 4
    selected_samples = np.random.choice(samples, num_samples, replace=False)
    # print(selected_samples)
    return samples, selected_samples

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


json_dict2=convert_dict(json_dict)

def get_box(name):
    data = json_dict2[name]
    return data


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










'''# Utility for reading an image and for getting its annotations.
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




    def visual(samples,new_dict):
        for file in samples:
            img=new_dict[file]["img_data"]
            imgplot = plt.imshow(img)
            plt.show()



    ##########visualize_few_examples(selected_samples,json_dict)
    #visual(selected_samples,json_dict2)'''