import keras.models
from model_definition import Classifier
from Configs import SharedConfigurations
from keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

conf=SharedConfigurations()
path_classes=conf.path_classes
batch_size=conf.batch_size
img_size=conf.img_size
val_ratio=conf.val_ratio
rnd_seed=conf.rnd_seed
img_mode=conf.img_mode

dataset_train = image_dataset_from_directory(path_classes,
                                                 batch_size=batch_size,
                                                 image_size=(img_size[0] * 2, img_size[1] * 2),  # img_size,
                                                 subset='training',
                                                 validation_split=val_ratio,
                                                 seed=rnd_seed,
                                                 color_mode=img_mode,
                                                 label_mode='categorical')

saved_models=conf.saved_models
aug_path=conf.augumented_examples_path
model_path=conf.path_model

def get_models(models_path):
    models=os.listdir(models_path)
    models_list=[]
    for model in models:
        #print(model)
        model_path=models_path + "\\" + model
        models_list.append(model_path)
    return models_list

models_list=get_models(saved_models)

def get_stamp(inp_str):
    num = ""
    for c in inp_str:
        if c.isdigit():
            num = num + c
    return num




def get_layer(model_path):
    model=Classifier.load_custom_model(model_path)
    aug_layer=model.get_layer(index=1)
    return aug_layer


def get_layer_output(layer):
    # layer.summary()
    inp=layer.input
    otp=layer.output
    get_layer_output_via_K = K.function([inp],  [otp])

    for images, _ in dataset_train.take(1):
        rnd_batch=images

    lay_otp=get_layer_output_via_K(rnd_batch)
    lay_otp=lay_otp[0]
    return lay_otp



def save_augumented_examples(saving_path,images,model_num):
    cnt=0
    for image in images:
        fig=plt.figure(figsize=(10, 10))
        fig.add_subplot(1,1,1)
        plt.imshow(image)
        plt.axis("off")
        image_name= saving_path + "\\" + "aug_" + model_num + "_" + str(cnt) + ".png" # add model number
        cnt+=1
        fig.savefig(image_name)   # save the figure to file
        plt.close(fig)

def get_augumented_images(models_list):
    for model in models_list:
        model_num=model[-6:]
        model_num=model_num[:-1]
        model_num = get_stamp(model_num)
        #print(model_num)
        aug_layer=get_layer(model)
        lay_otp = get_layer_output(aug_layer)
        save_augumented_examples(aug_path, lay_otp,model_num)  # add model number




get_augumented_images(models_list)