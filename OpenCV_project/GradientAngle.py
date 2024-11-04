import cv2
import numpy as np

def GradientAngle(inputimg):
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

    # magnitude image
    magnitude_image = np.zeros_like(blur).astype(np.int32)
    magnitude_image = np.sqrt(x_image**2 + y_image**2)

    angle = (np.arctan2(y_image, x_image) * 180 / np.pi) % 360 

    # mask
    mask1 = cv2.inRange(angle, 170, 190)
    mask2 = cv2.inRange(angle, 260, 280)

    masked_image1 = cv2.bitwise_and(magnitude_image, magnitude_image, mask=mask1)
    masked_image2 = cv2.bitwise_and(magnitude_image, magnitude_image, mask=mask2)

    # show image
    cv2.imshow("angle1",masked_image1.astype(np.uint8))
    cv2.imshow("angle2",masked_image2.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    