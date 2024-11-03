import cv2 
import numpy as np

def ColorSeperation(inputimg):
    inputimg = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)
    r,g,b = cv2.split(inputimg)

    # define zeros as blank image(2D array with all zeros)
    zeros = np.zeros(inputimg.shape[:2], dtype = "uint8")

    b_image = cv2.merge([b, zeros, zeros])
    g_image = cv2.merge([zeros, g, zeros])
    r_image = cv2.merge([zeros, zeros, r])

    # show 3 images
    cv2.imshow("Red", r_image)
    cv2.imshow("Green", g_image)
    cv2.imshow("Blue", b_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

