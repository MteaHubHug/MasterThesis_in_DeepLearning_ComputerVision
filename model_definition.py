import tensorflow.keras.models
import numpy as np
import tensorflow as tf

from typing import Tuple

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense,Dropout,GlobalAveragePooling2D,BatchNormalization

from tensorflow.keras.applications import efficientnet


from keras.layers import RandomFlip,RandomTranslation,RandomRotation,RandomZoom,Resizing


class RandomColor(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super(RandomColor, self).__init__(**kwargs)

        self._range_contrast   = ( 0.8, 1.2)
        self._range_gamma      = ( 0.9, 1,1)
        self._range_hue        = ( 0.9, 1.1)
        self._range_saturation = ( 0.8, 1.2)
        self._range_brightness = ( -0.2, 0.2)

    def call(self, images, training=None):

        if not training:
            return images

        contrast   = np.random.uniform(*self._range_contrast)
        gamma      = np.random.uniform(*self._range_gamma)
        hue        = np.random.uniform(*self._range_hue)
        saturation = np.random.uniform(*self._range_saturation)
        brightness = np.random.uniform(*self._range_brightness)

        images = tf.image.adjust_contrast(images,contrast)
        images = tf.image.adjust_gamma(images,gamma)
        images = tf.image.adjust_hue(images,hue)
        images = tf.image.adjust_saturation(images,saturation)
        images = tf.image.adjust_brightness(images, delta=brightness)

        return images

img_augmentation = Sequential(
    [

        RandomFlip(),
        RandomRotation(factor=0.02),
        RandomTranslation(height_factor=0.1,width_factor=0.1),
        RandomZoom(height_factor=[0.1,-0.2],width_factor=[0.1,-0.2]),
        RandomColor()
        #RandomContrast(factor=[0.3,0.0]),
        #Resizing(height=200, width=320)
    ],
    name='augmentor',
)

class Classifier:

    @staticmethod
    def build_model(img_hwc:Tuple[int,int,int], nr_classes:int)-> Model:


        img = Input(shape=(img_hwc[0]*2,img_hwc[1]*2,img_hwc[2]), name='image')
        augment = img_augmentation(img)
        resize = Resizing(height=img_hwc[0], width=img_hwc[1], name='resizing')(augment)

        base = efficientnet.EfficientNetB0(input_shape=img_hwc,
                                           include_top=False,
                                           #drop_connect_rate=0.4,
                                           weights='imagenet')(resize)

        predictor = GlobalAveragePooling2D(name='avgpool')(base) #either keep it or exchange it with another pooling layer
        #predictor = BatchNormalization(name='batchnorm')(predictor)
        #predictor = Dropout(0.4,name='drop')(predictor) # 0.3

        predictor = Dense(nr_classes,
                    activation='softmax',
                    name='probs')(predictor)

        return Model(inputs=img,
                     outputs=predictor,
                     name='classification')

    def load_custom_model(path_to_model:str)-> Model:
        model = tensorflow.keras.models.load_model(path_to_model, custom_objects={'RandomColor': RandomColor})

        return model
