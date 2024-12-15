import numpy as np
import matplotlib.pyplot as plt
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
import warnings
import sys
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

warnings.filterwarnings("ignore", category=DeprecationWarning)

class UI_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI_MainWindow, self).__init__()
        self.setWindowTitle("Opencvdl_hw2")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()
        self.img1 = None
        self.img2 = None

    def initUI(self):
        # button for load image 1
        self.btnimage1 = QtWidgets.QPushButton("Load Image1", self)
        self.btnimage1.setGeometry(50, 100, 150, 30)
        self.btnimage1.clicked.connect(self.loadImage1)

        # button for load image 2
        self.btnimage2 = QtWidgets.QPushButton("Load Image2", self)
        self.btnimage2.setGeometry(50, 500, 150, 30)
        self.btnimage2.clicked.connect(self.loadImage2)

        # button for task 1.1 show structure 01
        self.btntask11 = QtWidgets.QPushButton("1.1 Show Structure", self)
        self.btntask11.setGeometry(320, 100, 250, 30)
        self.btntask11.clicked.connect(self.ShowStructure01)

        # button for task 1.2 show acc and loss
        self.btntask12 = QtWidgets.QPushButton("1.2 Show Acc and Loss", self)
        self.btntask12.setGeometry(320, 180, 250, 30)
        self.btntask12.clicked.connect(self.ShowAccLoss)

        # button for task 1.3 Predict
        self.btntask13 = QtWidgets.QPushButton("1.3 Predict", self)
        self.btntask13.setGeometry(320, 260, 250, 30)
        self.btntask13.clicked.connect(self.Predict)

        # button for task 2.1 Show images
        self.btntask21 = QtWidgets.QPushButton("2.1 Show Images", self)
        self.btntask21.setGeometry(320, 500, 250, 30)
        self.btntask21.clicked.connect(self.ShowImages)

        # button for task 2.2 Show Structure 02
        self.btntask22 = QtWidgets.QPushButton("2.2 Show Model Structure", self)
        self.btntask22.setGeometry(320, 580, 250, 30)
        self.btntask22.clicked.connect(self.ShowStructure02)

        # button for task 2.3 Show comparison
        self.btntask23 = QtWidgets.QPushButton("2.3 Show Comparison", self)
        self.btntask23.setGeometry(320, 660, 250, 30)
        self.btntask23.clicked.connect(self.ShowComparison)

        # button for task 2.4 Inference
        self.btntask24 = QtWidgets.QPushButton("2.4 Inference", self)
        self.btntask24.setGeometry(320, 740, 250, 30)
        self.btntask24.clicked.connect(self.Inference)


        # label for image 1
        self.lblimage1 = QtWidgets.QLabel(self)
        self.lblimage1.setGeometry(650, 100, 320, 240)
        self.lblimage1.setStyleSheet("border: 3px solid black;")

        # label for image 2
        self.lblimage2 = QtWidgets.QLabel(self)
        self.lblimage2.setGeometry(650, 500, 480, 320)
        self.lblimage2.setStyleSheet("border: 3px solid black;")

    def loadImage1(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            self.img1 = cv2.imread(fileName)
            print(f"Image1: {fileName} loaded successfully")
            self.lblimage1.setPixmap(QtGui.QPixmap(fileName))
            self.lblimage1.setScaledContents(True)

    def loadImage2(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            self.img2 = cv2.imread(fileName)
            print(f"Image2: {fileName} loaded successfully")

    def ShowStructure01(self):
        pass

    def ShowAccLoss(self):
        pass

    def Predict(self):
        pass

    def ShowImages(self):
        pass

    def ShowStructure02(self):
        pass    

    def ShowComparison(self):
        pass

    def Inference(self):
        pass
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = UI_MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())