import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
from  PIL  import  Image
from tensorflow.keras.models import Model,Sequential
from keras.layers import RandomFlip,RandomTranslation,RandomRotation,RandomZoom,Resizing
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from Configs import SharedConfigurations

###############################################################################
conf=SharedConfigurations()
batch_size=conf.batch_size
img_size=conf.img_size
val_ratio=conf.val_ratio
rnd_seed=conf.rnd_seed
img_mode=conf.img_mode
path_classes=r"C:\Users\lukic4\Desktop\cameliaKut\usecase Nachverpacken"
dataset_train = image_dataset_from_directory(path_classes,
                                             batch_size=batch_size,
                                             image_size=(img_size[0] * 2, img_size[1] * 2),  # img_size,
                                             subset='training',
                                             validation_split=val_ratio,
                                             seed=rnd_seed,
                                             color_mode=img_mode,
                                             label_mode='categorical')

for images, _ in dataset_train.take(1):
    rnd_batch = images
my_img=rnd_batch[0] # just one image
######################################################################################################


range_contrast = random.uniform(0.8, 1.2)
range_gamma = random.uniform(0.9, 1.1)
range_hue=random.uniform(0.9,1.1)
range_saturation = random.uniform(0.8, 1.2)
range_brightness = random.uniform(-0.2, 0.2)

def random_color(image):


    #image = tf.image.adjust_contrast(image, range_contrast)
    #image = tf.image.adjust_gamma(image, range_gamma)
    image = tf.image.adjust_hue(image, range_hue)
    #image = tf.image.adjust_saturation(image, range_saturation)
    #image = tf.image.adjust_brightness(image, range_brightness)
    return image

filtered=random_color(my_img)

path_to_save=r"D:\filtered.png"
res=array_to_img(filtered)
save_img(path_to_save,res)

'''

img_augmentation = Sequential(
    [

        RandomFlip(),
        RandomRotation(factor=0.02),
        RandomTranslation(height_factor=0.1,width_factor=0.1),
        RandomZoom(height_factor=[0.1,-0.2],width_factor=[0.1,-0.2]),
        #RandomColor()
        #RandomContrast(factor=[0.3,0.0]),
        #Resizing(height=200, width=320)
    ],
    name='augmentor',
)

image = tf.expand_dims(im3, 0)
fig=plt.figure(figsize=(10, 10))


rows = 3
columns = 3

for i in range(9):
  augmented_image = img_augmentation(image)
  fig.add_subplot(rows, columns, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")
plt.show()'''