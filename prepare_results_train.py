import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix

from Configs import SharedConfigurations
from model_definition import Classifier


def prepare_and_save_plot(image_folder, image_height, image_width, model, image_name, results_folder, ground_truth_label, f_pred_ok_str, f_pred_nok_str):

    os.chdir(image_folder)

    impath = image_folder + "\\" + image_name
    im1 = cv2.imread(impath)
    model_input_image = np.expand_dims(cv2.resize(im1, (image_width, image_height)), 0)
    output = model(model_input_image)

    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])

    a = output.numpy()
    data = [a[0,1], a[0,0]]
    delta_confidence = abs(data[0] - data[1])
    bins = [0,1]

    ax = plt.subplot(1, 2, 2)

    if data[0] > data[1]:
        pred_str = f_pred_ok_str
    else:
        pred_str = f_pred_nok_str

    if pred_str == ground_truth_label:
        color_str = 'green'
    else:
        color_str = 'orange'


    ax.bar(bins, data, color=color_str, edgecolor='gray')

    ax.set_xticks(bins)

    for prob, x in zip(data, bins):
        # Label the percentages
        if prob > 0.5:
            ax.annotate(str(prob), xy=(x, prob), xycoords=('data', 'data'),
                        xytext=(0, prob - 18), textcoords='offset points', va='top', ha='center')
        else:
            ax.annotate(str(prob), xy=(x, prob), xycoords=('data', 'data'),
                        xytext=(0, prob + 18), textcoords='offset points', va='top', ha='center')

    # Make some more room at the bottom of the plot
    plt.subplots_adjust(bottom=0.15)
    plt.xlim([-0.5, 1.5])
    plt.ylim([0, 1])
    newxticks = ['OK', 'Nachverpacken']
    plt.xticks([0, 1], newxticks)
    plt.xlabel('Predicted Label')
    plt.ylabel('Probability')

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(25.6, 13.37)

    os.chdir(results_folder)

    image_name2 = f'score[{delta_confidence:.7f}]' + image_name
    plt.savefig(image_name2, dpi=300)

    plt.close()


if __name__ == '__main__':

    # load model

    configs = SharedConfigurations()

    orig_train_folder = configs.orig_train_folder
    orig_valid_folder = configs.orig_valid_folder

    path_model = configs.path_model
    model = Classifier.load_custom_model(path_to_model=path_model)

    results_train_folder = orig_train_folder + '\\results'
    results_valid_folder = orig_valid_folder + '\\results'

    # example_image = 'C:\\Users\\marcuzzi\\Desktop\\FMarcuzzi\\Wuerth\\TRAINING\\usecase_nachverpackung\\OK\\2101417008_2021_09_06_13_32_01_OK.jpg'

    # image transformation for model input
    height = model.input_shape[1]
    width  = model.input_shape[2]

    n_classes = model.output_shape[1]

    # string results for confusion matrix
    pred_ok_str  = configs.string_ok_prediction
    pred_nok_str = configs.string_not_ok_prediction

    ## analyze the training part of the dataset
    print('*' * 40)
    print('Analyzing the training part of the dataset')
    print('*' * 40)

    ground_truth_str = []
    predictions_str = []

    if os.getcwd() != orig_train_folder:
        os.chdir(orig_train_folder)

    # Examine OK ground truth cases
    image_folder_ok_training = orig_train_folder + '\\OK'
    os.chdir(image_folder_ok_training)
    for ctr, im_name in enumerate(os.listdir(image_folder_ok_training)):
        if np.mod(ctr,50) == 0:
            print('[Ground Truth OK cases] doing file: ' + str(ctr+1) + '/' + str(len(os.listdir(image_folder_ok_training))))

        impath= image_folder_ok_training + "\\" + im_name
        im1 = cv2.imread(impath)
        model_input_image = np.expand_dims(cv2.resize(im1, (width, height)), 0)
        output = model(model_input_image)

        ok_prob = output.numpy()[0,1]
        nok_prob = output.numpy()[0,0]

        ground_truth_str.append(pred_ok_str)

        if ok_prob > nok_prob:
            predictions_str.append(pred_ok_str)
        else:
            predictions_str.append(pred_nok_str)


    image_folder_nok_training = orig_train_folder + '\\NOK'
    os.chdir(image_folder_nok_training)
    for ctr, im_name in enumerate(os.listdir(image_folder_nok_training)):
        print('[Ground Truth US cases] doing file: ' + str(ctr + 1) + '/' + str(len(os.listdir(image_folder_nok_training))))

        impath = image_folder_nok_training + "\\" + im_name
        im1 = cv2.imread(impath)
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
    heatmap_name = 'Confusion_matrix_training_dataset.jpg'
    os.chdir(results_train_folder)
    plt.savefig(heatmap_name, dpi=300)
    plt.close()
    os.chdir(orig_train_folder)

    # analyze images and save them with histograms

    im_folders = [image_folder_ok_training, image_folder_nok_training]
    for i_f in im_folders:

        print('doing folder ' + i_f)

        if i_f == image_folder_ok_training:
            ground_truth_str = pred_ok_str
        elif i_f == image_folder_nok_training:
            ground_truth_str = pred_nok_str

        for ctr, img in enumerate(os.listdir(i_f)):

            print('done ' + str(ctr+1) + ' images')

            prepare_and_save_plot(i_f, height, width, model, img, results_train_folder, ground_truth_str, pred_ok_str, pred_nok_str)

