from PIL import Image
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
import json
import os
from Configs import SharedConfigurations
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt
import numpy as np
import keras
from Keypoint_detec_Generator import get_samples
conf = SharedConfigurations()

IMG_DIR=  r"D:\FINAL DATASET\wuerth_iriis_annotate"
TEST_DATASET_PATH = IMG_DIR

RESULTS_DIR = r"D:\Augumented_examples"
JSON = conf.IRIIS_json
results_path=conf.keypoint_detector_models_path
model_path= conf.keypoint_detec_model
chosen_models_path=conf.keypoint_detec_chosen_models

json_dict = json.load(JSON)

IMG_SIZE = conf.keypoint_detec_IMG_SIZE
BATCH_SIZE = len(os.listdir(IMG_DIR) ) #############
NUM_KEYPOINTS = conf.num_keypoints

def get_box(name, images_dict):
    data = images_dict[name]
    return data

def get_test_dict(json_dict,test_dir):
    cnt=0
    new_dict={}
    files=os.listdir(test_dir)
    for file in files:
        if(cnt%100==0): print(cnt)
        inner_dict={}
        filename="wuerth_iriis/" + file + "-1"
        regions = json_dict["_via_img_metadata"][filename]["regions"]
        if (len(regions)>0):
            xs = json_dict["_via_img_metadata"][filename]["regions"][0]["shape_attributes"]["all_points_x"]
            ys = json_dict["_via_img_metadata"][filename]["regions"][0]["shape_attributes"]["all_points_y"]

            p1 = [xs[0], ys[0], 1]
            p2 = [xs[1], ys[1], 1]
            p3 = [xs[2], ys[2], 1]
            p4 = [xs[3], ys[3], 1]
            keypoints = [p1, p2, p3, p4]

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
        cnt+=1
    return new_dict

test_dict=get_test_dict(json_dict,TEST_DATASET_PATH)


model = keras.models.load_model(model_path)
samples, selected_samples = get_samples(IMG_DIR)

np.random.shuffle(samples)
validation_keys = samples

train_aug = iaa.Sequential(
    [
        iaa.Resize(IMG_SIZE, interpolation="linear"),
        iaa.Fliplr(0.3),
        iaa.Flipud(0.15),
        iaa.AddToBrightness((-30, 30)),
        iaa.AddToHue((-50, 50)),
        iaa.GammaContrast((0.5, 2.0)),
        iaa.AddToSaturation((-50, 50)),
        # `Sometimes()` applies a function randomly to the inputs with
        # a given probability (0.3, in this case).
        iaa.Sometimes(0.4, iaa.Affine(rotate=(-45,45), scale=(0.5, 0.7))), # don't want to rotate it too much without scaling, might lose keypoints
    ]
)

def data_generation(batch_size,  keys ,images_dict, aug):
        batch_images = np.empty((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype="int")
        batch_keypoints = np.empty(
            (batch_size, 1, 1, NUM_KEYPOINTS), dtype="float32"
        )
        batch_keys=[]
        for i, key in enumerate(keys):
            batch_keys.append(key)
            data = get_box(key,images_dict)
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
            (new_image, new_kps_obj) =  aug(image=current_image, keypoints=kps_obj)
            batch_images[i,] = new_image
            # Parse the coordinates from the new keypoint object.
            kp_temp = []
            for keypoint in new_kps_obj:
                kp_temp.append(np.nan_to_num(keypoint.x))
                kp_temp.append(np.nan_to_num(keypoint.y))
            batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, 4 * 2)
        batch_keypoints = batch_keypoints / IMG_SIZE
        return (batch_images, batch_keypoints) , batch_keys


######################################################################################################
############################### AUGUMENTATION VISUALISATION ####################################################
######################################################################################################



(images, ground_truth_keypoints), batch_keys = data_generation(BATCH_SIZE,samples,test_dict,train_aug)
sample_val_images= images
sample_ground_truth_keypoints =  ground_truth_keypoints.reshape(-1, 4, 2) * IMG_SIZE
#sample_ground_truth_keypoints=next(iter(ground_truth_keypoints))
#sample_ground_truth_keypoints = sample_ground_truth_keypoints.reshape(-1, 4, 2) * IMG_SIZE


def visual_augumentations(samples,save_dir, keys, keypoints):
    i=0
    cnt=0
    for img in samples:
        imname= save_dir + "\\" + keys[i]
        print(imname)
        p1 = keypoints[i][0]
        p2 = keypoints[i][1]
        p3 = keypoints[i][2]
        p4 = keypoints[i][3]
        plt.plot(p1[0], p1[1], marker='v', color="green")
        plt.plot(p2[0], p2[1], marker='v', color="green")
        plt.plot(p3[0], p3[1], marker='v', color="green")
        plt.plot(p4[0], p4[1], marker='v', color="green")
        plt.imshow(img)
        #plt.show()

        plt.savefig(imname,dpi=300)
        plt.close()
        i+=1

        cnt+=1
        if(cnt==15):
            return

visual_augumentations(sample_val_images, RESULTS_DIR ,batch_keys, sample_ground_truth_keypoints)

