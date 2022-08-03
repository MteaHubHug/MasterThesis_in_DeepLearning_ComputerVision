import os
from keras.preprocessing.image import load_img, save_img
import cv2
from matplotlib import pyplot as plt
cropped_images_dir=r"C:\compare_res"


def rotate_sirius(folder):
    images=os.listdir(folder)
    for image in images:
        if(image[-9:]=="color.png"):
            imname=folder + "\\" + image
            img=cv2.imread(imname)
            h, w, _ = img.shape
            img= cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img= cv2.resize(img, (w,h))
            cv2.imwrite(imname,img)

#rotate_sirius(cropped_images_dir)

def make_graphs(folder):
    images=os.listdir(folder)

    for j in range(0,len(images),2):
        plt.figure(figsize=(10, 10))  # specifying the overall grid size
        z=0
        for i in range(j,j+2):
            imname=folder + "\\" + "hist_" + images[i]
            #img=cv2.imread(folder + "\\" + images[i])
            img= load_img(folder + "\\" + images[i])
            plt.subplot(1, 2, z + 1)
            plt.title(images[i])
            z+=1
            plt.imshow(img)
        plt.savefig(imname)

make_graphs(cropped_images_dir)