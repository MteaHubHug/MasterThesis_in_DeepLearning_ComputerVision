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

config =SharedConfigurations()
path_train=config.DEPTH_ESTIMATOR_WHOLE_DATASET
#path_train=r"E:\DEPTH_ESTIMATOR\DATASET_train"
#path_validation=r"E:\DEPTH_ESTIMATOR\DATASET_validation"
path_result_models=config.DEPTH_ESTIMATOR_RESULTS_MODELS
save_dir=config.DEPTH_ESTIMATOR_RESULTS_DEPTHS
HEIGHT = 256
WIDTH = 256
LR = 0.0002
BATCH_SIZE = 128
#### get data from folder ##########################
def get_data(path):
    filelist = []
    files=os.listdir(path )
    for file in files:
        #if(file[-4:]!="krdi"):
            #print(file)
            file_path=path  + "\\" + file
            filelist.append(file_path)
    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith("iriis.jpg")],
        "depth": [x for x in filelist if x.endswith("depth.png")],
        #"mask": [x for x in filelist if x.endswith("-depth_mask.npy")],
    }
    return data

data=get_data(path_train) #path_validation
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42)

########### build  pipeline ###########################

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=6, dim=(768, 1024), n_channels=3, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        filenames=[]
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, image_path, depth_path ):
        """Load input and target image."""

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)


        depth_map = cv2.imread(depth_path)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def data_generation(self, batch):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load(
                self.data["image"][batch_id],
                self.data["depth"][batch_id],
                #self.data["mask"][batch_id],
            )
        return x, y


def visualize_depth_map(save_dir,df,samples, test=False, model=None):
    input, target = samples
    cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
    cmap.set_bad(color="black")

    if test:
        pred = model.predict(input)
        #fig, ax = plt.subplots(6, 3, figsize=(50, 50))
        for i in range(4):
            imname = df._get_value(i, "image", takeable=False)
            imname= imname.split("\\")[3]
            imname= save_dir + "\\" + imname
            print(imname)
            #imname= save_dir + "\\" + filenames[i]
            #print(imname)
            #plt.imshow((input[i].squeeze()))
            #plt.show()
            #plt.imshow((target[i].squeeze()), cmap=cmap)
            #plt.show()

            plt.imshow((pred[i].squeeze()), cmap=cmap)
            ###plt.show()
            plt.axis("off")
            plt.savefig(imname,dpi=300)
            plt.close()
    else:
        #fig, ax = plt.subplots(6, 2, figsize=(50, 50))
        for i in range(4):
            plt.imshow((input[i].squeeze()))
            plt.show()
            plt.imshow((target[i].squeeze()), cmap=cmap)
            plt.show()


#data_gen=DataGenerator(data=df, batch_size=6, dim=(HEIGHT, WIDTH))
#visualize_samples  = next(iter(data_gen))

#visualize_depth_map(save_dir,df,visualize_samples)
