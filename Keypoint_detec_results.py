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

IMG_DIR=  r"D:\FINAL DATASET\wuerth_iriis_theRest"
####IMG_DIR = conf.not_annotated_IRIIS_images_folder
TEST_DATASET_PATH = IMG_DIR

RESULTS_DIR= conf.keypoint_detec_results_path
JSON = conf.IRIISandSIRIUS_json
results_path=conf.keypoint_detector_models_path  ############
model_path= conf.keypoint_detec_model

json_dict = json.load(JSON)

IMG_SIZE = conf.keypoint_detec_IMG_SIZE
BATCH_SIZE = len(os.listdir(IMG_DIR) ) #############
NUM_KEYPOINTS = conf.num_keypoints

def get_box(name, images_dict):
    if name in images_dict:
       data = images_dict[name]
       return data
    else:
       print(" File : ", name, " is not in the JSON dictionary")
       return -1

def get_image_keys(json_dict):
    keys=json_dict["_via_image_id_list"]
    all_keys={}
    for key in keys:
        orig_key=key
        if(key[0]=="w"):
            key=key[13:]
            key=key[:-2]
        else:
            key=key[:-7]
        #print(key, " *** ", orig_key)
        all_keys[key]=orig_key
    return all_keys

def get_test_dict(json_dict,test_dir):
    cnt=0
    new_dict={}
    files=os.listdir(test_dir)
    keys=get_image_keys(json_dict)
    for file in files:
        inner_dict = {}
        if file in keys:
            if(cnt%100==0): print(cnt)
            filename=keys[file]
            regions = json_dict["_via_img_metadata"][filename]["regions"]
            if (len(regions)==0):
                p1 = [0, 0, 0]
                keypoints = [p1, p1, p1, p1]

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
test_aug = iaa.Sequential([iaa.Resize(IMG_SIZE, interpolation="linear")])

def test_data_generation(batch_size,  keys ,images_dict, aug):
        batch_images = np.empty((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype="int")
        batch_keypoints = np.empty(
            (batch_size, 1, 1, NUM_KEYPOINTS), dtype="float32"
        )
        batch_keys=[]
        for i, key in enumerate(keys):
            batch_keys.append(key)
            data = get_box(key,images_dict)
            if (data!= -1):
                current_keypoint = np.array(data["joints"])[:, :2]
                kps = []
                for j in range(0, len(current_keypoint)):
                    kps.append(Keypoint(x=current_keypoint[j][0], y=current_keypoint[j][1]))
                current_image = data["img_data"]
                kps_obj = KeypointsOnImage(kps, shape=current_image.shape)
                (new_image, new_kps_obj) =  aug(image=current_image, keypoints=kps_obj)
                batch_images[i,] = new_image
                kp_temp = []
                for keypoint in new_kps_obj:
                    kp_temp.append(np.nan_to_num(keypoint.x))
                    kp_temp.append(np.nan_to_num(keypoint.y))
                batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, 4 * 2)
                batch_keypoints = batch_keypoints / IMG_SIZE
        return (batch_images, batch_keypoints) , batch_keys

validation_dataset , batch_keys = test_data_generation(BATCH_SIZE,samples,test_dict,test_aug )
sample_val_images = next(iter(validation_dataset))
predictions = model.predict(sample_val_images).reshape(-1, 4, 2) * IMG_SIZE

def visual_results(samples,keypoints,save_dir, keys):
    i=0
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
        plt.show()

        plt.savefig(imname,dpi=300)
        plt.close()
        i+=1


def save_result_keypoints_in_json(samples,keypoints,save_dir, keys):
    i=0
    regions={}
    results=[]
    for img in samples:
        imname= save_dir + "\\" + keys[i]
        print(imname)
        p1 = keypoints[i][0]
        p2 = keypoints[i][1]
        p3 = keypoints[i][2]
        p4 = keypoints[i][3]

        xs=[   str(p1[0])  ,   str(p2[0])    ,   str(p3[0])    ,    str(p4[0])    ]
        ys=[   str(p1[1])  ,   str(p2[1])    ,   str(p3[1])    ,    str(p4[1])    ]
        regions["filename"]=keys[i]
        regions["all_points_x"]=xs
        regions["all_points_y"]=ys
        results.append(regions)
        i+=1
    json_object = json.dumps(results, indent=4)
    with open("Keypoints_results.json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()

#visual_results(sample_val_images,predictions, RESULTS_DIR ,batch_keys  )
save_result_keypoints_in_json(sample_val_images,predictions, RESULTS_DIR ,batch_keys  )

''' " "regions": [
        {
          "shape_attributes": {
            "name": "polyline",
            "all_points_x": [
              117,
              1197,
              1190,
              99
            ],
            "all_points_y": [
              62,
              49,
              730,
              746
            ]
          },
          "region_attributes": {}
        }
      ]'''