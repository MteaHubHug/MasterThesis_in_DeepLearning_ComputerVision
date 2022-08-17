import os
import shutil
import copy
import matplotlib
import keras.models
import pandas as pd
from matplotlib import pyplot as plt
from Depth_Estimator_Generator import DataGenerator, visualize_depth_map, get_data
from Configs import SharedConfigurations
conf=SharedConfigurations()
path_test_data=r"E:\DEPTH_ESTIMATOR\DATASET_test"
HEIGHT=conf.DEPTH_ESTIMATOR_HEIGHT
WIDTH=conf.DEPTH_ESTIMATOR_WIDTH
results_dir=conf.DEPTH_ESTIMATOR_RESULTS_DEPTHS
model_path=conf.DEPTH_ESTIMATOR_MODEL
model=keras.models.load_model(model_path)

data=get_data(path_test_data) #path_validation
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42)

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
            plt.axis("off")
            ###plt.show()
            plt.savefig(imname,dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
    else:
        #fig, ax = plt.subplots(6, 2, figsize=(50, 50))
        for i in range(4):
            plt.imshow((input[i].squeeze()))
            plt.show()
            plt.imshow((target[i].squeeze()), cmap=cmap)
            plt.show()

test_loader = next(
    iter(
        DataGenerator(
            data=df, batch_size=265, dim=(HEIGHT, WIDTH) # 265
        )
    )
)
visualize_depth_map(results_dir,df, test_loader, test=True, model=model)


