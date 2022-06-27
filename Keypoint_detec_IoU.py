import math

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
import shapely
from shapely.geos import TopologicalError
from shapely.geometry import Polygon
from Keypoint_detec_Generator import get_samples
conf = SharedConfigurations()

######################### NAME OF THE SCRIPT IS IoU, but we are using it for getting average Eucledean Distance

IMG_DIR=  r"D:\FINAL DATASET\wuerth_iriis_annotate"
####IMG_DIR = conf.not_annotated_IRIIS_images_folder
TEST_DATASET_PATH = IMG_DIR

#RESULTS_DIR= conf.keypoint_detec_results_path
RESULTS_DIR = r"D:\IoU"
JSON = conf.IRIIS_json
results_path=conf.keypoint_detector_models_path
model_path= conf.keypoint_detec_model
chosen_models_path=conf.keypoint_detec_chosen_models

json_dict = json.load(JSON)

IMG_SIZE = conf.keypoint_detec_IMG_SIZE
BATCH_SIZE = len(os.listdir(IMG_DIR) ) #############
AMOUNT = BATCH_SIZE
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
#predictions = model.predict(sample_val_images).reshape(-1, 4, 2) * IMG_SIZE

######################################################################################################
####################################### RESULTS PREPERATION FOR IoU#####################################################
######################################################################################################

(images, ground_truth_keypoints), batch_keys1 = test_data_generation(BATCH_SIZE,samples,test_dict,test_aug)
sample_ground_truth_keypoints =  ground_truth_keypoints.reshape(-1, 4, 2) * IMG_SIZE
#sample_ground_truth_keypoints=next(iter(ground_truth_keypoints))
#sample_ground_truth_keypoints = sample_ground_truth_keypoints.reshape(-1, 4, 2) * IMG_SIZE


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            # union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 1
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def avg_distance(list1, list2):
    """
    Eucledean distance between ground-truth keypoints and predicted
    """
    Euc_dist=0
    dist1=math.dist(list1[0],list2[0])
    dist2=math.dist(list1[1],list2[1])
    dist3=math.dist(list1[2],list2[2])
    dist4=math.dist(list1[3],list2[3])
    Euc_dist = dist1 + dist2 + dist3 + dist4
    Euc_dist = Euc_dist / 4
    return Euc_dist

def get_results(samples,keypoints, keys,ground_truth_keypoints, save_dir):
    i=0
    cnt=0
    results=[]
    IoU_sum=0
    EucDist_sum=0
    for img in samples:
        imname= save_dir + "\\" + keys[i]
        #print(imname)
        p1 = keypoints[i][0]
        p2 = keypoints[i][1]
        p3 = keypoints[i][2]
        p4 = keypoints[i][3]
        predictions=[p1,p2,p3,p4]



        g1 = ground_truth_keypoints[i][0]
        g2 = ground_truth_keypoints[i][1]
        g3 = ground_truth_keypoints[i][2]
        g4 = ground_truth_keypoints[i][3]
        ground_truths=[g1,g2,g3,g4]

        ##########IoU=polygon_iou(ground_truths,predictions)
        ##########IoU_sum+=IoU

        EucDist= avg_distance(ground_truths,predictions)
        EucDist_sum+=EucDist

        plt.plot(p1[0], p1[1], marker='v', color="green")
        plt.plot(p2[0], p2[1], marker='v', color="green")
        plt.plot(p3[0], p3[1], marker='v', color="green")
        plt.plot(p4[0], p4[1], marker='v', color="green")

        plt.plot(g1[0], g1[1], marker='v', color="pink")
        plt.plot(g2[0], g2[1], marker='v', color="pink")
        plt.plot(g3[0], g3[1], marker='v', color="pink")
        plt.plot(g4[0], g4[1], marker='v', color="pink")
        #plt.imshow(img)
        ###################plt.show()

        #plt.savefig(imname,dpi=300)
        #plt.close()

        #print("*********************")
        #print(ground_truths)
        #print(predictions)
        #print(IoU)
        #print("*********************")

        i+=1

        cnt += 1
        if (cnt == AMOUNT):
            return EucDist_sum


#IoU_sum=get_results(sample_val_images,predictions,batch_keys, sample_ground_truth_keypoints ,RESULTS_DIR)

def compare_models(models_path, sample_val_images, batch_keys, sample_groundtruth_keypoints, RESULTS_DIR):
    models=os.listdir(models_path)
    IoU_sums={}
    EucDist_sums={}
    for m in models:
        print("checking model : ", m)
        model_path=models_path + "\\" + m
        model = keras.models.load_model(model_path)
        predictions = model.predict(sample_val_images).reshape(-1, 4, 2) * IMG_SIZE
        #IoU_sum = get_results(sample_val_images, predictions, batch_keys, sample_groundtruth_keypoints, RESULTS_DIR)
        #avg_IoU_m = IoU_sum / AMOUNT

        Eucledean_dist_sum=get_results(sample_val_images, predictions, batch_keys, sample_groundtruth_keypoints, RESULTS_DIR)
        avg_Eucledean_dist= Eucledean_dist_sum / AMOUNT
        #print("Average IoU for model : ", m , " is : ", avg_IoU_m)
        print("Average Eucledean distance for model : ", m , " is ", avg_Eucledean_dist)
        #IoU_sums[m]=IoU_sum
        EucDist_sums[m]=Eucledean_dist_sum

    #best_model = max(IoU_sums, key=IoU_sums.get)
    best_model2 = min(EucDist_sums, key=EucDist_sums.get)
    #best_IoU=  IoU_sums[best_model]
    #avg_IoU= best_IoU/  AMOUNT

    best_EucDist= EucDist_sums[best_model2]
    print(" Best model is : ", best_model2)
    #print(" avg IoU of the best model : ", avg_IoU)

    #print(" avg Eucledean Distance : ", avg_Eucledean_dist)

compare_models(chosen_models_path, sample_val_images,batch_keys,sample_ground_truth_keypoints,RESULTS_DIR)