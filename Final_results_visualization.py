import os

from PIL import Image
from matplotlib import pyplot as plt
from Configs import SharedConfigurations


conf=SharedConfigurations()
folder=r"E:\NEURAL_IMAGE_ENHANCER_FINAL_RESULTS"


def make_plot(folder):
    images=os.listdir(folder)
    plt.figure(figsize=(10, 10))  # specifying the overall grid size

    for i in range(9):
        plt.subplot(3, 3, i + 1)  # the number of images in the grid is 5*5 (25)
        imname=folder + "\\" + images[i]
        img=Image.open(imname)
        plt.imshow(img)
        plt.axis("off")
    plt.axis('off')
    plt.show()

make_plot(folder)