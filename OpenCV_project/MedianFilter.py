import cv2
import numpy as np

image = None

def MedianFilter(inputimg):
    global image
    image = inputimg
    cv2.namedWindow("Median Filter")
    cv2.createTrackbar("m", "Median Filter", 0, 5, apply_median_filter)

    apply_median_filter(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_median_filter(x):
    m = cv2.getTrackbarPos("m", "Median Filter")
    kernel = 2*m+1
    blurred_image = cv2.medianBlur(image, kernel)
    cv2.imshow("Median Filter", blurred_image)
