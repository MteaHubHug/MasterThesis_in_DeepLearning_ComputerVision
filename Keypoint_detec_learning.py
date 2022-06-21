
from Configs import SharedConfigurations
from tensorflow import keras
import imgaug.augmenters as iaa
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import os
from Keypoint_detec_model_definition import get_model
from Keypoint_detec_Generator import KeyPointsDataset
from Keypoint_detec_Generator import get_samples
from matplotlib import pyplot as plt
conf = SharedConfigurations()

IMG_SIZE = conf.keypoint_detec_IMG_SIZE
BATCH_SIZE = conf.keypoint_detec_BATCH_SIZE
EPOCHS = conf.keypoint_detec_EPOCHS
NUM_KEYPOINTS = conf.num_keypoints

IMG_DIR = conf.annotated_IRIIS_images_folder
JSON = conf.IRIIS_json

samples, selected_samples = get_samples(IMG_DIR)

train_aug = iaa.Sequential(
    [
        iaa.Resize(IMG_SIZE, interpolation="linear"),
        iaa.Fliplr(0.3),
        # `Sometimes()` applies a function randomly to the inputs with
        # a given probability (0.3, in this case).
        iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7))),
    ]
)
test_aug = iaa.Sequential([iaa.Resize(IMG_SIZE, interpolation="linear")])

np.random.shuffle(samples)
train_keys, validation_keys = (
    samples[int(len(samples) * 0.15):],
    samples[: int(len(samples) * 0.15)],
)

train_dataset = KeyPointsDataset(train_keys, train_aug)
validation_dataset = KeyPointsDataset(validation_keys, test_aug, train=False)

print(f"Total batches in training set: {len(train_dataset)}")
print(f"Total batches in validation set: {len(validation_dataset)}")

sample_images, sample_keypoints = next(iter(train_dataset))
sample_keypoints = sample_keypoints[:4].reshape(-1, 4, 2) * IMG_SIZE
path_results = conf.keypoint_detector_models_path

## model build ######
callbacks = [
    ModelCheckpoint(os.path.join(path_results, 'keypoint_detector/save_at_{epoch}.h5')),
    TensorBoard(log_dir=os.path.join(path_results, 'tensorboard'))
]
get_model().summary()
model = get_model()
model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-4), metrics=['accuracy'])
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=callbacks)


## learning curves : ####

def get_learning_curves(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()
    plt.close()

get_learning_curves(history)

## predictions : ######

#sample_val_images, sample_val_keypoints = next(iter(validation_dataset))
#sample_val_images = sample_val_images[:4]
#sample_val_keypoints = sample_val_keypoints[:4].reshape(-1, 4, 2) * IMG_SIZE
#predictions = model.predict(sample_val_images).reshape(-1, 4, 2) * IMG_SIZE
