import sys, cv2 
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLineEdit
from qt_material import apply_stylesheet
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class UI_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI_MainWindow, self).__init__()
        self.setWindowTitle("OpenCV with PyQt5")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):

        # label for background image
        self.label_background = QtWidgets.QLabel(self)
        self.label_background.setGeometry(QtCore.QRect(0, -450, 1920, 1920))
        self.label_background.setObjectName("label_3")
        self.label_background.setStyleSheet("image: url(./background.png);")

        # button for load image 1
        self.btnimage1 = QtWidgets.QPushButton("Load Image1", self)
        self.btnimage1.setGeometry(50, 440, 150, 30)
        self.btnimage1.clicked.connect(self.loadImage1)

        # button for load image 2
        self.btnimage2 = QtWidgets.QPushButton("Load Image2", self)
        self.btnimage2.setGeometry(50, 520, 150, 30)
        self.btnimage2.clicked.connect(self.loadImage2)

        # button for task 1.1 Color separation
        self.btntask11 = QtWidgets.QPushButton("1.1 Color Separation", self)
        self.btntask11.setGeometry(320, 100, 250, 30)
        self.btntask11.clicked.connect(self.ColorSeparation)
        self.btntask11.setStyleSheet("border: 3px solid cyan;")

        # button for task 1.2 Color transformation
        self.btntask12 = QtWidgets.QPushButton("1.2 Color Transformation", self)
        self.btntask12.setGeometry(320, 180, 250, 30)
        self.btntask12.clicked.connect(self.ColorTransformation)
        self.btntask12.setStyleSheet("border: 3px solid cyan;")

        # button for task 1.3 Color Extraction
        self.btntask13 = QtWidgets.QPushButton("1.3 Color Extraction", self)
        self.btntask13.setGeometry(320, 260, 250, 30)
        self.btntask13.clicked.connect(self.ColorExtraction)
        self.btntask13.setStyleSheet("border: 3px solid cyan;")
                                       
        # button for task 2.1 Gaussian blur
        self.btntask21 = QtWidgets.QPushButton("2.1 Gaussian Blur", self)
        self.btntask21.setGeometry(320, 400, 250, 30)
        self.btntask21.clicked.connect(self.GaussianBlur)
        self.btntask21.setStyleSheet("border: 3px solid cyan;")

        # button for task 2.2 Bilateral filter
        self.btntask22 = QtWidgets.QPushButton("2.2 Bilateral Filter", self)
        self.btntask22.setGeometry(320, 480, 250, 30)
        self.btntask22.clicked.connect(self.BilateralFilter)
        self.btntask22.setStyleSheet("border: 3px solid cyan;")

        # button for task 2.3 Median filter
        self.btntask23 = QtWidgets.QPushButton("2.3 Median Filter", self)
        self.btntask23.setGeometry(320, 560, 250, 30)
        self.btntask23.clicked.connect(self.MedianFilter)
        self.btntask23.setStyleSheet("border: 3px solid cyan;")

        # button for task 3.1 Sobel X
        self.btntask31 = QtWidgets.QPushButton("3.1 Sobel X", self)
        self.btntask31.setGeometry(320, 700, 250, 30)
        self.btntask31.clicked.connect(self.SobelX)
        self.btntask31.setStyleSheet("border: 3px solid cyan;")

        # button for task 3.2 Sobel Y
        self.btntask32 = QtWidgets.QPushButton("3.2 Sobel Y", self)
        self.btntask32.setGeometry(320, 780, 250, 30)
        self.btntask32.clicked.connect(self.SobelY)
        self.btntask32.setStyleSheet("border: 3px solid cyan;")

        # button for task 3.3 Combination and Threshold
        self.btntask33 = QtWidgets.QPushButton("3.3 Combination and Threshold", self)
        self.btntask33.setGeometry(320, 860, 250, 30)
        self.btntask33.clicked.connect(self.CombinationThreshold)
        self.btntask33.setStyleSheet("border: 3px solid cyan;")

        # button for task 3.4 Gradient Angle
        self.btntask34 = QtWidgets.QPushButton("3.4 Gradient Angle", self)
        self.btntask34.setGeometry(320, 940, 250, 30)
        self.btntask34.clicked.connect(self.GradientAngle)
        self.btntask34.setStyleSheet("border: 3px solid cyan;")

        # label for Rotation degree
        self.label_rotation = QtWidgets.QLabel(self)
        self.label_rotation.setGeometry(1020, 100, 800, 30)
        self.label_rotation.setText("Rotation:                                               deg")  
        self.label_rotation.setStyleSheet("font-size: 18px; font-weight: bold; color: cyan;")

        # label for Scaling factor
        self.label_scaling = QtWidgets.QLabel(self)
        self.label_scaling.setGeometry(1020, 200, 800, 30)
        self.label_scaling.setText("Scaling:")
        self.label_scaling.setStyleSheet("font-size: 18px; font-weight: bold; color: cyan;")

        # label for Tx
        self.label_tx = QtWidgets.QLabel(self)
        self.label_tx.setGeometry(1020, 300, 800, 30)
        self.label_tx.setText("Tx:                                                            pixel")
        self.label_tx.setStyleSheet("font-size: 18px; font-weight: bold; color: cyan;")

        # label for Ty
        self.label_ty = QtWidgets.QLabel(self)
        self.label_ty.setGeometry(1020, 400, 800, 30)
        self.label_ty.setText("Ty:                                                            pixel")
        self.label_ty.setStyleSheet("font-size: 18px; font-weight: bold; color: cyan;")

        # line edit to input rotation degree
        self.text_rotation = QtWidgets.QLineEdit(self)
        self.text_rotation.setGeometry(1140, 100, 100, 30)
        self.text_rotation.setReadOnly(False)
        self.text_rotation.setEnabled(True)
        self.text_rotation.setStyleSheet("border: 2px solid cyan; font-size: 18px;")

        # line edit to input scaling factor
        self.text_scaling = QtWidgets.QLineEdit(self)
        self.text_scaling.setGeometry(1140, 200, 100, 30)
        self.text_scaling.setReadOnly(False)
        self.text_scaling.setEnabled(True)
        self.text_scaling.setStyleSheet("border: 2px solid cyan; font-size: 18px;")

        # line edit to input Tx
        self.text_tx = QtWidgets.QLineEdit(self)
        self.text_tx.setGeometry(1140, 300, 100, 30)
        self.text_tx.setReadOnly(False)
        self.text_tx.setEnabled(True)
        self.text_tx.setStyleSheet("border: 2px solid cyan; font-size: 18px;")
        
        # line edit to input Ty
        self.text_ty = QtWidgets.QLineEdit(self)
        self.text_ty.setGeometry(1140, 400, 100, 30)
        self.text_ty.setReadOnly(False)
        self.text_ty.setEnabled(True)
        self.text_ty.setStyleSheet("border: 2px solid cyan; font-size: 18px;")

        # button to perform Transforms
        self.btn_transform = QtWidgets.QPushButton("Transforms", self)
        self.btn_transform.setGeometry(1090, 500, 200, 30)
        self.btn_transform.setStyleSheet("border: 3px solid cyan;")
        self.btn_transform.clicked.connect(self.Transforms)



        # label for question 1, image processing
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(300, 50, 200, 30)
        self.label.setText("1. Image Processing")
        self.label.setStyleSheet("font-size: 18px; font-weight: bold; color: cyan;")

        # label for question 2, Image Smoothing
        self.label2 = QtWidgets.QLabel(self)
        self.label2.setGeometry(300, 350, 200, 30)
        self.label2.setText("2. Image Smoothing")
        self.label2.setStyleSheet("font-size: 18px; font-weight: bold; color: cyan;")

        # label for question 3, Edge Detection
        self.label3 = QtWidgets.QLabel(self)
        self.label3.setGeometry(300, 650, 200, 30)
        self.label3.setText("3. Edge Detection")
        self.label3.setStyleSheet("font-size: 18px; font-weight: bold; color: cyan;")

        # label for question 4, Transforms
        self.label4 = QtWidgets.QLabel(self)
        self.label4.setGeometry(1000, 50, 200, 30)
        self.label4.setText("4. Transforms")
        self.label4.setStyleSheet("font-size: 18px; font-weight: bold; color: cyan;")


        # label frame for question 1
        self.label_frame = QtWidgets.QLabel(self)
        self.label_frame.setGeometry(300, 80, 300, 230)
        self.label_frame.setStyleSheet("border: 3px solid cyan;")   

        # label frame for question 2
        self.label_frame2 = QtWidgets.QLabel(self)
        self.label_frame2.setGeometry(300, 380, 300, 230)
        self.label_frame2.setStyleSheet("border: 3px solid cyan;")

        # label frame for question 3
        self.label_frame3 = QtWidgets.QLabel(self)
        self.label_frame3.setGeometry(300, 680, 300, 310)
        self.label_frame3.setStyleSheet("border: 3px solid cyan;")

        # label frame for question 4
        self.label_frame4 = QtWidgets.QLabel(self)
        self.label_frame4.setGeometry(1000, 80, 400, 500)
        self.label_frame4.setStyleSheet("border: 3px solid cyan;")

        self.text_rotation.raise_()
        self.text_scaling.raise_()
        self.text_tx.raise_()
        self.text_ty.raise_()
        self.btntask11.raise_()
        self.btntask12.raise_()
        self.btntask13.raise_()
        self.btntask21.raise_()
        self.btntask22.raise_()
        self.btntask23.raise_()
        self.btntask31.raise_()
        self.btntask32.raise_()
        self.btntask33.raise_()
        self.btntask34.raise_()
        self.btn_transform.raise_()
 
    def loadImage1(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            img1 = cv2.imread(fileName)
            print(f"Image1: {fileName} loaded successfully")


    def loadImage2(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            img2 = cv2.imread(fileName)
            print(f"Image2: {fileName} loaded successfully")

    def ColorSeparation(self):
        pass
    
    def ColorTransformation(self):
        pass

    def ColorExtraction(self):
        pass

    def GaussianBlur(self):
        pass

    def BilateralFilter(self):
        pass

    def MedianFilter(self):
        pass

    def SobelX(self):
        pass

    def SobelY(self):
        pass

    def CombinationThreshold(self):
        pass

    def GradientAngle(self):
        pass

    def Transforms(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    window = UI_MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())