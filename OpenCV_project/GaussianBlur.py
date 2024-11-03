import cv2
import numpy as np

image = None

def GaussianBlur(inputimg):
    global image
    image = inputimg
    cv2.namedWindow("Gaussian Blur")
    cv2.createTrackbar("m", "Gaussian Blur", 0, 5, apply_gaussian_blur)

    apply_gaussian_blur(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def apply_gaussian_blur(x):
    m = cv2.getTrackbarPos("m", "Gaussian Blur")
    kernel = 2*m+1
    SigmaX = ((kernel-1)/2 -1)*0.3+0.8
    SigmaY = ((kernel-1)/2 -1)*0.3+0.8
    blurred_image = cv2.GaussianBlur(image, (kernel, kernel), SigmaX, SigmaY)
    cv2.imshow("Gaussian Blur", blurred_image)


