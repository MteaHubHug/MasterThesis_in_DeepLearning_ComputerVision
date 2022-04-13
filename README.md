# Introduction 
Neural Image Enhancer is a Neural Network model that takes an RGB image as an input, and predicts informations such as depth, volume or distance.
The idea is to use this model to calculate the volume of an empty space in a box. It is useful because then boxes can be reduced or filled with more content such as adding material if it is needed.
    

# Getting Started
 Guide :
1.	Installation process
For edditing scripts, I used PyCharm as dev environment. Language is Python. Libraries and packages are listed in "requirements.txt". 
For training, I am using Docker Container tensorflow:2.6.0-gpu
 ( https://hub.docker.com/layers/tensorflow/tensorflow/tensorflow/2.6.0-gpu/images/sha256-e0510bc8ea7dfed3b1381f92891fae3ad5b75712984c3632a60cc164391bca29?context=explore )


2. Software dependencies
Linux OS

3. Hardware dependencies
GPU


# Build and Test
USECASES : Nachverpacken & Umschlichtung
Firstly, we want to have preparated dataset. 
For that purpose, it is needed to set the path in Configs.py. After that, we are ready to run scripts "extract_krdi_files.py" and "prepare_dataset.py".
All steps are described in comments in those scripts.

After preparing the data, we should have dataset organized in folders for each usecase. 
In a folder of usecase, there should be folders that corresponding to classes. 
For example, folder "Usecase Nachverpacken" should contain folders "OK" and "NOK" 
(in order to use keras function "Image_dataset_from_directory" for loading the datasets for training/validation).

All hyperparameters should be defined in "Configs.py" (and paths, of course).

For training, build the docker container and install all the requirments from "requirements.txt". 
After that, just run the script "learn_from_scratch.py" and follow learing curves on tensorboard. 
When you see that accuracy and loss are satisfying,
 you can stop the training and use "model_saved_at_epoch.h5" 
 (epoch which has best accuracy and loss) for predicting the images. 

Model is defined in "model_definition.py"

For visualizing the results, organize dataset with "separate_train_valid_data.py", then use "prepare_results_validation.py".
