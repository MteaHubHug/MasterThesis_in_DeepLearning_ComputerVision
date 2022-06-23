import os
import numpy as np
import tensorflow as tf
import math
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils import class_weight

from Configs import SharedConfigurations
from model_definition import Classifier


if __name__ == '__main__':

    print('running classifier')

    # path to resources

    config = SharedConfigurations()

    path_classes = config.path_classes
    path_results = config.path_training_results

    # parameters for training

    val_ratio = config.val_ratio
    img_size = config.img_size
    img_mode = config.img_mode
    batch_size = config.batch_size
    rnd_seed = config.rnd_seed

    callbacks = [
        ModelCheckpoint(os.path.join(path_results, 'keras_models/save_at_{epoch}.h5')),
        TensorBoard(log_dir=os.path.join(path_results, 'tensorboard'))
    ]

    dataset_train = image_dataset_from_directory(path_classes,
                                                 batch_size=batch_size,
                                                 image_size=(img_size[0] * 2, img_size[1] * 2),  # img_size,
                                                 subset='training',
                                                 validation_split=val_ratio,
                                                 seed=rnd_seed,
                                                 color_mode=img_mode,
                                                 label_mode='categorical')

    class_names = dataset_train.class_names

    dataset_valid = image_dataset_from_directory(path_classes,
                                                 batch_size=batch_size,
                                                 image_size=(img_size[0] * 2, img_size[1] * 2),  # img_size,
                                                 subset='validation',
                                                 validation_split=val_ratio,
                                                 seed=rnd_seed,
                                                 color_mode=img_mode,
                                                 label_mode='categorical')




    trainfile = open(config.train_images_name_file, 'w')
    for namestr in dataset_train.file_paths:
        trainfile.write(namestr + '\n')
    trainfile.close()

    validfile = open(config.validation_images_name_file, 'w')
    for namestr in dataset_valid.file_paths:
        validfile.write(namestr + '\n')
    validfile.close()

    strategy = tf.distribute.MirroredStrategy()  # strategy is used for training on machines with multiple GPUs. Not
    # specifying a device in the constructor argument will prompt the
    # program to use all available GPUs.

    number_of_training_files = len(dataset_train.file_paths)
    number_of_validation_files = len(dataset_valid.file_paths)

    # we want to have ~100k iterations overall. We split the training into two phases. The first 150 epochs are run only
    # with the added layers unlocked for training. This should give the model enough time to saturate to best
    # performance. After this, the final layers of the efficientnet are unlocked and the model is trained until we reach
    # the max number of iterations.

    tot_iterations = config.tot_iterations
    iterations_per_epoch = math.ceil(number_of_training_files / batch_size)
    tot_epochs = math.ceil(tot_iterations / iterations_per_epoch)
    epochs_base_locked = round(tot_epochs/2)
    epochs_base_unlocked = round(tot_epochs/2)

    labels = ['OK'] * number_of_training_files + ['NOK'] * number_of_validation_files
    class_weights_dict = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = {idx: w for idx, w in zip(range(len(class_weights_dict)), class_weights_dict)}

    base_id = 3
    with strategy.scope():
        model = Classifier.build_model((img_size[0], img_size[1], 1 if img_mode == 'grayscale' else 3), len(class_names))

        # set the whole efficient net part as fixed, no training for the first 'epochs_base_locked' epochs
        model.layers[base_id].trainable = False

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()

    model.fit(dataset_train,
              epochs=epochs_base_locked,
              callbacks=callbacks,
              validation_data=dataset_valid,
              workers=10,
              use_multiprocessing=True)

    # after the first 'epochs_base_locked' epochs, we switch to unlock some of the final layers of the efficientnet

    with strategy.scope():

        model.layers[base_id].trainable = True
        for ctr,layer in enumerate(model.layers[base_id].layers):
            if not isinstance(layer,BatchNormalization) and ctr >= 221: # change this ==> find the one that is close to the end
                layer.trainable = True
            else:
                layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()

    model.fit(dataset_train,
              epochs=epochs_base_unlocked,
              callbacks=callbacks,
              validation_data=dataset_valid,
              class_weight=class_weights_dict)

    print('DONE')
