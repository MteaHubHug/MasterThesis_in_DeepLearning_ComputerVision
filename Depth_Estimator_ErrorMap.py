import cv2
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2

img1=cv2.imread("ref1.png",0)
height, width = img1.shape
img2=cv2.imread("ref2.jpg",0)
img2=cv2.resize(img2,(width,height))
import cv2

import cv2

'''# Load images as grayscale

# Calculate the per-element absolute difference between
# two arrays or between an array and a scalar
diff = 255 - cv2.absdiff(img1, img2)
cv2.imshow('diff', diff)
cv2.waitKey()'''



grayA = img1
grayB = img2

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# show the output images
cv2.imshow("Original", img1)
cv2.imshow("Predicted", img2)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)


