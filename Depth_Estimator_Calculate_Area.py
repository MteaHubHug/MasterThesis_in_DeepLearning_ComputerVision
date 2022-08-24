import cv2
from PIL import Image
import numpy as np
from matplotlib import  pyplot as plt

def calculate_area(image_path):
    img=cv2.imread(image_path)
    #cv2.imshow("img",img)
    #cv2.waitKey()
    number_of_low_pixels=0
    for a in img:
        for b in a:
            distance=0
            for c in b:
                distance+=c
            distance=distance//3  # distnace in centimeters
            print(distance)
            if distance< 40:
                number_of_low_pixels+=1
    print(number_of_low_pixels)
    if(number_of_low_pixels>85000):
        print("Nachverpacken")
    else:
        print("OK")

# if one pixel is one milimeter
# calculate the volume
def max_volume(image_path):
    img=cv2.imread(image_path)
    h, w, channels= img.shape
    a= h/10
    b= w/10
    volume= a * b * 50
    return volume

def calculate_volume(image_path,max_volume):
    img=cv2.imread(image_path)
    #cv2.imshow("img",img)
    #cv2.waitKey()
    h, w, channels= img.shape
    edge1= h/10
    edge2= w/10
    distances=0
    for a in img:
        for b in a:
            distance=0
            for c in b:
                distance+=c
            distance=distance//3
            distances+=distance
    avg_distances= distances / int(h*w)
    volume= edge1 * edge2 * avg_distances
    avg_volume=volume / max_volume
    if(avg_volume<80):
        print("Nachverpacken , box is  : ", round(avg_volume * 100,2), " % full")
    else :
        print("OK , box is  : ", round(avg_volume * 100,2), " % full")




image = r"E:\NEURAL_IMAGE_ENHANCER_FINAL_RESULTS\normalization\2101416172_20210906T130215_sirius-depth-cropnorm.png"
#calculate_area(image)
the_volume=max_volume(image)
calculate_volume(image,the_volume)
