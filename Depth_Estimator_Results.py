import os
import shutil
import copy
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
from matplotlib.pylab import cm
import keras.models
import pandas as pd
from matplotlib import pyplot as plt
from Configs import SharedConfigurations
conf=SharedConfigurations()
#path_test_data=r"E:\DEPTH_ESTIMATOR\DATASET_test"
path_test_data=r"E:\NEURAL_IMAGE_ENHANCER_FINAL_RESULTS\proba"
HEIGHT=conf.DEPTH_ESTIMATOR_HEIGHT
WIDTH=conf.DEPTH_ESTIMATOR_WIDTH
#results_dir=conf.DEPTH_ESTIMATOR_RESULTS_DEPTHS
results_dir=r"E:\NEURAL_IMAGE_ENHANCER_FINAL_RESULTS\res_2"
model_path=conf.DEPTH_ESTIMATOR_MODEL
model=keras.models.load_model(model_path)
IMG_SIZE=224

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
        "image": [x for x in filelist if x.endswith("iriis-crop.jpg")],
        #"depth": [x for x in filelist if x.endswith("depth-crop.png")],
        #"mask": [x for x in filelist if x.endswith("-depth_mask.npy")],
    }
    return data

data=get_data(path_test_data) #path_validation
df = pd.DataFrame(data)
#df = df.sample(frac=1, random_state=42)
df = df.sample(frac=1)
def visualize_depth_map(img_dir,save_dir,df,samples, test=False, model=None):
    input = samples
    cmap = copy.copy(cm.get_cmap("jet"))
    cmap.set_bad(color="black")
    w = 244
    h = 244
    if test:
        pred = model.predict(input)
        #fig, ax = plt.subplots(6, 3, figsize=(50, 50))
        for i in range(len(img_dir)):
            imname = df._get_value(i, "image", takeable=False)
            imname= imname.split("\\")[3][:-4]
            imname= save_dir + "\\" + imname + "_estim.jpg"
            #print(imname)
            #imname= save_dir + "\\" + filenames[i]
            #print(imname)
            #plt.imshow((input[i].squeeze()))
            #plt.show()
            #plt.imshow((target[i].squeeze()), cmap=cmap)
            #plt.show()

            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow((pred[i].squeeze()),cmap=cmap)
            fig.savefig(imname, dpi=1)

            #plt.imshow((pred[i].squeeze()), cmap=cmap,figsize=(2.44, 2.44),dpi=100)
            #plt.axis("off")
            ###plt.show()
            #plt.savefig(imname,dpi=100, bbox_inches='tight', pad_inches=0)
            #plt.close()




class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=10, dim=(224, 224), n_channels=3, shuffle=True):
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
        x= self.data_generation(batch)

        return x

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, image_path ):
        """Load input and target image."""

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)



        return image_

    def data_generation(self, batch):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        for i, batch_id in enumerate(batch):
            x[i,] = self.load(
                self.data["image"][batch_id]
            )
        return x



test_loader = next(
    iter(
        DataGenerator(
            data=df, batch_size=50, dim=(HEIGHT, WIDTH) # 265
        )
    )
)
visualize_depth_map(path_test_data,results_dir,df, test_loader, test=True, model=model)


