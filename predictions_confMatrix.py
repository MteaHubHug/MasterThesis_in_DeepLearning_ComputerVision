import os
import numpy as np
import cv2
from Configs import SharedConfigurations
from model_definition import Classifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

configs = SharedConfigurations()
path_model = configs.path_model
model = Classifier.load_custom_model(path_to_model=path_model)

orig_valid_folder = configs.orig_valid_folder

results_valid_folder = orig_valid_folder + '\\results'

height = model.input_shape[1]
width = model.input_shape[2]

n_classes = model.output_shape[1]

pred_class1_str = configs.string_ok_prediction
pred_class2_str = configs.string_not_ok_prediction


## analyze the validation part of the dataset
print('*' * 40)
print('Analyzing the validation part of the dataset')
print('*' * 40)

ground_truth_str = []
predictions_str = []

if os.getcwd() != orig_valid_folder:
    os.chdir(orig_valid_folder)



def get_predictions(stringClass,intClassNum):
    i=intClassNum+1
    i_str=str(i)
    image_folder_class_validation = orig_valid_folder + "\\"+stringClass
    preds= []
    ground_truths=[]
    os.chdir(image_folder_class_validation)
    for ctr, im_name in enumerate(os.listdir(image_folder_class_validation)):
        if np.mod(ctr, 50) == 0:
            print('[Ground Truth CLASS' + stringClass + '  cases] doing file: ' + str(ctr + 1) + '/' + str(
                len(os.listdir(image_folder_class_validation))))

        im1 = cv2.imread(im_name)
        model_input_image = np.expand_dims(cv2.resize(im1, (width, height)), 0)
        output = model(model_input_image)

        prob = output.numpy()[0]
        pred = np.argmax(prob)
        preds.append(pred)
        ground_truths.append(intClassNum)
    return [ground_truths,preds]
##############################################################

def get_inputs_for_conf_matrix(NumberOfClasses):
    ground_truths=[]
    predictions=[]
    for i in range(0,NumberOfClasses):
        i_str=str(i)
        pred=get_predictions(i_str,i)
        #print(pred)
        g_t=pred[0]
        preds=pred[1]
        ground_truths.extend(g_t)
        predictions.extend(preds)
    return ground_truths, predictions

###########################################################

def make_conf_matrix(ground_truths, predictions,results_folder):
    print("****************")
    print(ground_truths)
    print("****************")
    print(predictions)
    print("****************")
    print("Making ConFusion Matrix ...")
    labels = ["OK","NOK"] ############### ????????????????????????????????? [ 0, 1] ?? [1,2]
    C = confusion_matrix(ground_truths, predictions, labels=labels)
    sns.heatmap(C, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels, annot=True, fmt='d')
    heatmap_name = results_folder + "\\" + 'Confusion_matrix_validation_dataset.jpg'
    plt.savefig(heatmap_name, dpi=300)
    plt.close()
    print("Confusion Matrix is saved to Results folder")


ground_truths, predictions= get_inputs_for_conf_matrix(6)
make_conf_matrix(ground_truths,predictions,results_valid_folder)
