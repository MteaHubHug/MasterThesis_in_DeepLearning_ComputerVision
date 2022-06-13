
###################

import os
import numpy as np
import cv2
from Configs import SharedConfigurations
from model_definition import Classifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    configs = SharedConfigurations()
    path_model = configs.path_model
    model = Classifier.load_custom_model(path_to_model=path_model)

    test_folder = configs.orig_valid_folder

    results_test_folder = test_folder + '\\results'

    height = model.input_shape[1]  # = 500
    width = model.input_shape[2]   # = 800


    n_classes = model.output_shape[1]

    # string results for confusion matrix
    pred_ok_str = configs.string_ok_prediction
    pred_nok_str = configs.string_not_ok_prediction

    ## analyze the validation part of the dataset
    print('*' * 40)
    print('Analyzing the validation part of the dataset')
    print('*' * 40)

    ground_truth_str = []
    predictions_str = []

    if os.getcwd() != test_folder:
        os.chdir(test_folder)

    # Examine OK ground truth cases
    image_folder_ok_test = test_folder + '\\OK'
    os.chdir(image_folder_ok_test)
    for ctr, im_name in enumerate(os.listdir(image_folder_ok_test)):
        if np.mod(ctr, 50) == 0:
            print('[Ground Truth OK cases] doing file: ' + str(ctr + 1) + '/' + str(
                len(os.listdir(image_folder_ok_test))))

        im1 = cv2.imread(im_name)
        model_input_image = np.expand_dims(cv2.resize(im1, (width, height)), 0)
        output = model(model_input_image)

        ok_prob = output.numpy()[0, 1]
        nok_prob = output.numpy()[0, 0]

        ground_truth_str.append(pred_ok_str)

        if ok_prob > nok_prob:
            predictions_str.append(pred_ok_str)
        else:
            predictions_str.append(pred_nok_str)

    image_folder_nok_test = test_folder + '\\NOK'
    os.chdir(image_folder_nok_test)
    for ctr, im_name in enumerate(os.listdir(image_folder_nok_test)):
        print('[Ground Truth US cases] doing file: ' + str(ctr + 1) + '/' + str(
            len(os.listdir(image_folder_nok_test))))
        im1 = cv2.imread(im_name)
        model_input_image = np.expand_dims(cv2.resize(im1, (width, height)), 0)
        output = model(model_input_image)

        ok_prob = output.numpy()[0, 1]
        nok_prob = output.numpy()[0, 0]

        ground_truth_str.append(pred_nok_str)

        if ok_prob > nok_prob:
            predictions_str.append(pred_ok_str)
        else:
            predictions_str.append(pred_nok_str)

    labels = [pred_ok_str, pred_nok_str]
    C = confusion_matrix(ground_truth_str, predictions_str, labels=labels)
    sns.heatmap(C, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels, annot=True, fmt='d')
    heatmap_name = 'Confusion_matrix_validation_dataset.jpg'
    os.chdir(results_test_folder)
    plt.savefig(heatmap_name, dpi=300)
    plt.close()
    os.chdir(test_folder)