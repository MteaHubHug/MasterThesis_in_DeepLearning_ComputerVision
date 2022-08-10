import os
import shutil

import keras.models
import pandas as pd
from Depth_Estimator_Generator import DataGenerator, visualize_depth_map, get_data
from Configs import SharedConfigurations
conf=SharedConfigurations()
path_test_data=r"E:\DEPTH_ESTIMATOR\DATASET_validation"
HEIGHT=conf.DEPTH_ESTIMATOR_HEIGHT
WIDTH=conf.DEPTH_ESTIMATOR_WIDTH
results_dir=conf.DEPTH_ESTIMATOR_RESULTS_DEPTHS
model_path=conf.DEPTH_ESTIMATOR_MODEL
model=keras.models.load_model(model_path)

data=get_data(path_test_data) #path_validation
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42)


test_loader = next(
    iter(
        DataGenerator(
            data=df, batch_size=265, dim=(HEIGHT, WIDTH) # 265
        )
    )
)
visualize_depth_map(results_dir,df, test_loader, test=True, model=model)


