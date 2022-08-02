
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import cv2
import matplotlib
import copy
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from Configs import SharedConfigurations
from Depth_Estimator_Model import DepthEstimationModel
from Depth_Estimator_Generator import DataGenerator, visualize_depth_map,df
config =SharedConfigurations()
path_train=r"E:\DEPTH_ESTIMATOR\DATASET_train"
path_validation=r"E:\DEPTH_ESTIMATOR\DATASET_validation"
path_result_models=config.DEPTH_ESTIMATOR_RESULTS_MODELS
results_dir=config.DEPTH_ESTIMATOR_RESULTS_DEPTHS
HEIGHT = 256
WIDTH = 256
LR = 0.0002
EPOCHS = 2
BATCH_SIZE = 32

###################### TRAINING ##############################
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR,
    amsgrad=False,
)

callbacks = [
    ModelCheckpoint(os.path.join(path_result_models, 'depth_estimator/DepthEstimator_save_at_{epoch}'),save_format="tf"),
    TensorBoard(log_dir=os.path.join(path_result_models, 'tensorboard'))
]

model = DepthEstimationModel()
# Define the loss function
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
# Compile the model
model.compile(optimizer, loss=cross_entropy)

train_loader = DataGenerator(
    data=df[:20].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH) # insted of 20 => should be 260
)
validation_loader = DataGenerator(
    data=df[20:].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH) # insted of 20 => should be 260
)
history=model.fit(
    train_loader,
    epochs=EPOCHS,
    validation_data=validation_loader,
    callbacks=callbacks
)
#################################################################################

test_loader = next(
    iter(
        DataGenerator(
            data=df, batch_size=6, dim=(HEIGHT, WIDTH) # 265
        )
    )
)
visualize_depth_map(results_dir,df, test_loader, test=True, model=model)

def get_learning_curves(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('depth_estimator_loss.png')
    plt.show()
    plt.close()

get_learning_curves(history)