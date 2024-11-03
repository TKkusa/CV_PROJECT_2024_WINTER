import cv2 
import numpy as np

def ColorTransformation(inputimg):
    inputimg = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)
    r,g,b = cv2.split(inputimg)

    # define zeros as blank image(2D array with all zeros)
    zeros = np.zeros(inputimg.shape[:2], dtype = "uint8")

    b_image = cv2.merge([b, zeros, zeros])
    g_image = cv2.merge([zeros, g, zeros])
    r_image = cv2.merge([zeros, zeros, r])

    cv_gray = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
    avg_gray = (b/3 + g/3 + r/3).astype(np.uint8)

    # show cv_gray and avg_gray
    cv2.imshow("cv_gray", cv_gray)
    cv2.imshow("avg_gray", avg_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
