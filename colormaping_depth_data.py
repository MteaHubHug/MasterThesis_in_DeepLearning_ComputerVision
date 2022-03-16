import os
import cv2

triplet_path = r"G:\Matea\triplets"
colorized_depths_path= r"G:\Matea\colorized_depths"

def use_colormap_on_depths(path1,path2):
    triplets = os.listdir(path1)
    os.chdir(path2)
    for image in triplets:
        if (image[-9:] == "depth.png"):
            grey_img_path = path1+ "\\" + image
            img = cv2.imread(grey_img_path)
            im1 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            new_name = path2+ "\\" + image
            cv2.imwrite(new_name, im1)

use_colormap_on_depths(triplet_path,colorized_depths_path)
