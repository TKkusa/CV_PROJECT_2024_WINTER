import cv2
import numpy as np

image = None

def BilateralFilter(inputimg):
    global image
    image = inputimg
    cv2.namedWindow("Bilateral Filter")
    cv2.createTrackbar("m", "Bilateral Filter", 0, 5, apply_bilateral_filter)

    apply_bilateral_filter(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_bilateral_filter(x):
    m = cv2.getTrackbarPos("m", "Bilateral Filter")
    kernel = 2*m+1
    blurred_image = cv2.bilateralFilter(image, kernel, 90, 90)
    cv2.imshow("Bilateral Filter", blurred_image)
