import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

orig_depth_folder=r"E:\KEYPOINT_DETECTOR\DEPTH_ESTIMATOR\SIRIUS_depths"
numpy_depth_folder="E:\KEYPOINT_DETECTOR\DEPTH_ESTIMATOR\SIRIUS_depths_numpy"



def convert_to_numpy(orig_folder,dest_folder):
    depths=os.listdir(orig_folder)
    for depth in depths:
        filename=orig_folder + "\\" + depth
        imname=dest_folder + "\\" + depth[:-4]
        img = Image.open(filename)
        data = np.array(img, dtype='uint8')
        np.save(imname + '.npy', data)

convert_to_numpy(orig_depth_folder,numpy_depth_folder)