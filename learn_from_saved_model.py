import os
import numpy as np
import tensorflow as tf
import math
import sklearn
import tensorflow.keras.optimizers

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

    configs = SharedConfigurations()

    path_classes = configs.path_classes
    path_results = configs.path_training_results  # error : path_training_results ?
    path_model = configs.path_model

    # parameters for training

    val_ratio = configs.val_ratio
    img_size = configs.img_size
    img_mode = configs.img_mode
    batch_size = configs.batch_size
    rnd_seed = configs.rnd_seed

    callbacks = [
        ModelCheckpoint(os.path.join(path_results,'keras_models/save_at_{epoch}.h5')),
        TensorBoard(log_dir=os.path.join(path_results,'tensorboard'))
    ]


    dataset_train = image_dataset_from_directory(path_classes,
                                                 batch_size=batch_size,
                                                 image_size=(img_size[0]*2,img_size[1]*2),#img_size,
                                                 subset='training',
                                                 validation_split=val_ratio,
                                                 seed=rnd_seed,
                                                 color_mode=img_mode,
                                                 label_mode='categorical')

    class_names = dataset_train.class_names

    dataset_valid = image_dataset_from_directory(path_classes,
                                                 batch_size=batch_size,
                                                 image_size=(img_size[0]*2,img_size[1]*2),#img_size,
                                                 subset='validation',
                                                 validation_split=val_ratio,
                                                 seed=rnd_seed,
                                                 color_mode=img_mode,
                                                 label_mode='categorical')

    trainfile = open(configs.train_images_name_file, 'w')
    for namestr in dataset_train.file_paths:
        trainfile.write(namestr + '\n')
    trainfile.close()

    validfile = open(configs.validation_images_name_file, 'w')
    for namestr in dataset_valid.file_paths:
        validfile.write(namestr + '\n')
    validfile.close()

    strategy = tf.distribute.MirroredStrategy() # strategy is used for training on machines with multiple GPUs. Not
                                                # specifying a device in the constructor argument will prompt the
                                                # program to use all available GPUs.


    number_of_training_files = len(dataset_train.file_paths)
    number_of_validation_files = len(dataset_valid.file_paths)

    # we want to have ~100k iterations overall. We split the training into two phases. The first 150 epochs are run only
    # with the added layers unlocked for training. This should give the model enough time to saturate to best
    # performance. After this, the final layers of the efficientnet are unlocked and the model is trained until we reach
    # the max number of iterations.

    tot_iterations = configs.tot_iterations
    iterations_per_epoch = math.ceil(number_of_training_files / batch_size)
    tot_epochs = math.ceil(tot_iterations / iterations_per_epoch)
    epochs_base_unlocked = tot_epochs

    labels = ['OK']*number_of_training_files + ['NOK']*number_of_validation_files
    class_weights_dict = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = {idx: w for idx, w in zip(range(len(class_weights_dict)), class_weights_dict)}

    base_id = 3
    with strategy.scope():
        model = Classifier.load_custom_model(path_to_model=path_model)

        model.layers[base_id].trainable = True

        model.layers[base_id].trainable = True
        for ctr, layer in enumerate(model.layers[base_id].layers):
            if not isinstance(layer, BatchNormalization) and ctr >= 221:
                layer.trainable = True
            else:
                layer.trainable = False

        model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()

    model.fit(dataset_train,
              epochs=epochs_base_unlocked,
              callbacks=callbacks,
              validation_data=dataset_valid,
              class_weight=class_weights_dict)


    print('DONE')




