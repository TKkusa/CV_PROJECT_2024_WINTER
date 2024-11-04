import cv2
import numpy as np

def SobelY(inputimg):
    gray_image = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
    Sigma = ((3-1)/2 -1)*0.3+0.8
    blur = cv2.GaussianBlur(gray_image, (3,3), Sigma, Sigma)

    # Sobel Y filter
    y_image_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    y_image = np.zeros_like(blur).astype(np.int32)

    width, height = y_image.shape

    for w in range(width):
        for h in range(height):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if w+i >= 0 and w+i < width and h+j >= 0 and h+j < height:
                        y_image[w, h] += blur[w+i, h+j] * y_image_filter[i+1, j+1]

    y_image = np.where(y_image < 0, y_image*-1, y_image)
    y_image = np.where(y_image > 255, 255, y_image)
    
    # show image
    cv2.imshow("Sobel_X",y_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()