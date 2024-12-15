import cv2
import numpy as np

def CombinationThreshold(inputimg):
    gray_image = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
    Sigma = ((3-1)/2 -1)*0.3+0.8
    blur = cv2.GaussianBlur(gray_image, (3,3), Sigma, Sigma)

    # Sobel X filter
    x_image_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    x_image = np.zeros_like(blur).astype(np.int32)

    width, height = x_image.shape
    for w in range(width):
        for h in range(height):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if w+i >= 0 and w+i < width and h+j >= 0 and h+j < height:
                        x_image[w, h] += blur[w+i, h+j] * x_image_filter[i+1, j+1]

    x_image = np.where(x_image < 0, x_image*-1, x_image)
    x_image = np.where(x_image > 255, 255, x_image)

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

    # Combination Threshold
    comb_image = np.zeros_like(blur).astype(np.int32)
    comb_image = np.sqrt(x_image**2 + y_image**2)

    # normalize comb_image
    comb_image = cv2.normalize(comb_image, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold with 128 and 28
    _, result1 = cv2.threshold(comb_image, 128, 255, cv2.THRESH_BINARY)
    _, result2 = cv2.threshold(comb_image, 28, 255, cv2.THRESH_BINARY)

    # show image
    cv2.imshow("Combination_Threshold",comb_image.astype(np.uint8))
    cv2.imshow("Threshold=128",result1.astype(np.uint8))
    cv2.imshow("Threshold=28",result2.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()