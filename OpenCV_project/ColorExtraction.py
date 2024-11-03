import cv2
import numpy as np

def ColorExtraction(inputimg):
    hsv_image = cv2.cvtColor(inputimg, cv2.COLOR_BGR2HSV)

    # The lower bound of the yellow-green HSV range
    lower_yellow_green = np.array([18, 0, 0])

    # The upper bound of the yellow-green HSV range
    upper_yellow_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv_image, lower_yellow_green, upper_yellow_green)
    mask_inverse = cv2.bitwise_not(mask)

    extracted_image = cv2.bitwise_and(inputimg, inputimg, mask = mask_inverse)

    # show extracted_image
    cv2.imshow("Extracted Image", extracted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    


