import os

import cv2
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2
depths_folder=r"E:\NEURAL_IMAGE_ENHANCER_FINAL_RESULTS\depths_orig_estim"
dest_folder=r"E:\NEURAL_IMAGE_ENHANCER_FINAL_RESULTS\ssims"
def save_ssims(img_folder,dest_folder):
    images=os.listdir(img_folder)
    for i in range(0,len(images),2):
        filename1=img_folder+"\\" + images[i]
        filename2=img_folder+"\\" + images[i+1]
        img1 = cv2.imread(filename1, 0)
        height, width = img1.shape
        img2 = cv2.imread(filename2, 0)
        img2 = cv2.resize(img2, (width, height))

        grayA = img1
        grayB = img2

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        text= images[i].split("_")[0] +"_" + images[i].split("_")[1]
        print( text, "{}".format(score))

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        diffname=dest_folder + "\\" + images[i][:-4] + "-ssimDiff.png"
        threshname=dest_folder + "\\" + images[i][:-4] + "-thresh.png"
        # show the output images
        #cv2.imshow("Original", img1)
        #cv2.imshow("Predicted", img2)
        saved1=cv2.imwrite(diffname,diff)
        saved2=cv2.imwrite(threshname,thresh)
        #cv2.imshow("Diff", diff)
        #cv2.imshow("Thresh", thresh)
        #cv2.waitKey(0)


def save_diffs(img_folder,dest_folder):
    images=os.listdir(img_folder)
    for i in range(0,len(images),2):
        filename1=img_folder+"\\" + images[i]
        filename2=img_folder+"\\" + images[i+1]
        img1 = cv2.imread(filename1, 0)
        height, width = img1.shape
        img2 = cv2.imread(filename2, 0)
        img2 = cv2.resize(img2, (width, height))

        diff = 255 - cv2.absdiff(img1, img2)
        #cv2.imshow('diff', diff)
        #cv2.waitKey()

        diffname=dest_folder + "\\" + images[i][:-4] + "-Diff.png"

        # show the output images
        #cv2.imshow("Original", img1)
        #cv2.imshow("Predicted", img2)
        saved=cv2.imwrite(diffname,diff)

        #cv2.imshow("Diff", diff)
        #cv2.imshow("Thresh", thresh)
        #cv2.waitKey(0)


save_ssims(depths_folder,dest_folder)
#save_diffs(depths_folder,dest_folder)

'''# Load images as grayscale

# Calculate the per-element absolute difference between
# two arrays or between an array and a scalar
diff = 255 - cv2.absdiff(img1, img2)
cv2.imshow('diff', diff)
cv2.waitKey()
'''