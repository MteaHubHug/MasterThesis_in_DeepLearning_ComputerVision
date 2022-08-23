import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.pylab import colorbar,cm
path=r"E:\NEURAL_IMAGE_ENHANCER_FINAL_RESULTS\provjera"
image_path=r"E:\NEURAL_IMAGE_ENHANCER_FINAL_RESULTS\provjera\2101416874_20210906T113837_naida.jpg"
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
def get_values_in_cm(image_path):
    import matplotlib.pyplot as plt
    import numpy as np
    #cmap=cm.get_cmap("jet")
    #norm = colors.Normalize(vmin=0, vmax=255)

    img=plt.imread(image_path)
    plt.imshow(img)
    plt.axis("off")
    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    #plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

    norm = plt.Normalize(0, 50)
    colorlist = ["darkorange","orange", "gold","yellow", "lawngreen","lightgreen"]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    plt.imshow(img, cmap=newcmp, norm=norm)
    plt.colorbar(cax=cax)

    plt.show()

#get_values_in_cm(image_path)



img=cv2.imread(image_path)

def normalize_image(image):
    for a in image:
        for b in a:
            for c in b:
                norma=c//5 -1
                print(c, norma)

normalize_image(img)


# 255 : 5 = 51

# 0-5 ==> 0
# 5-10 ==> 1
# 10-15 ==> 2
# 15-20 ==> 3
# ...
# 250 ==> 49
# 255 ==> 50