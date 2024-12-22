import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms, datasets
from torchsummary import summary
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale (1 channel)
    transforms.Resize((32, 32)),  # Resize to match the model input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Apply the same normalization as training
])

transform02 = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 requires 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet50 expects normalized inputs
])

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory containing inference images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform02
        self.image_paths = []  # List to store image file paths
        
        # Load all image paths from subfolders (Cat and Dog)
        for subfolder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    self.image_paths.append(os.path.join(folder_path, img_file))

    def __len__(self):
        # Return the total number of images
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image path
        img_path = self.image_paths[idx]
        
        # Load the image
        image = Image.open(img_path).convert('RGB')  # Ensure RGB mode
        
        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)
        
        return image, img_path  # Return image and its path for reference

class VGG16_BN(nn.Module):
    def __init__(self, num_classes = 10):
        super(VGG16_BN, self).__init__()
        self.vgg16 = models.vgg16_bn(pretrained=False)
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16(x)
    
class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
          param.requires_grad_ = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Modify the final layer for 2 classes
        self.model.fc.requires_grad_ = True

    def forward(self, x):
        return self.model(x)

class UI_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI_MainWindow, self).__init__()
        self.setWindowTitle("Opencvdl_hw2")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()
        self.img1 = None
        self.img2 = None
        self.img1path = ""
        self.img2path = ""

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
        self.lblimage1.setGeometry(650, 100, 55, 55)
        self.lblimage1.setStyleSheet("border: 3px solid black;")

        # label for image 2
        self.lblimage2 = QtWidgets.QLabel(self)
        self.lblimage2.setGeometry(650, 500, 224, 224)
        self.lblimage2.setStyleSheet("border: 3px solid black;")

        # label for predict result of image2
        self.lblpredict = QtWidgets.QLabel(self)
        self.lblpredict.setGeometry(650, 600, 480, 320)
        self.lblpredict.setText("Prediction: ")
        self.lblpredict.setStyleSheet("font-size: 18px;")
        

    def loadImage1(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        self.img1path = fileName
        print(fileName)
        if fileName:
            self.img1 = cv2.imread(fileName)
            print(f"Image1: {fileName} loaded successfully")
            self.lblimage1.setPixmap(QtGui.QPixmap(fileName))
            self.lblimage1.setScaledContents(True)

    def loadImage2(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        self.img2path = fileName
        if fileName:
            self.img2 = cv2.imread(fileName)
            print(f"Image2: {fileName} loaded successfully")
            self.lblimage2.setPixmap(QtGui.QPixmap(fileName))
            self.lblimage2.setScaledContents(True)

    def ShowStructure01(self):
        model = VGG16_BN().to(device)
        model.load_state_dict(torch.load("./model/vgg16_mnist.pth", map_location=device, weights_only=True))
        model.eval()
        summary(model, (1, 32, 32))

    
                              

    def ShowAccLoss(self):
        # show the training results png file
        img = cv2.imread("training_results.png")
        cv2.imshow("Training Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def Predict(self):
        if self.img1path == "":
            print("Please load an image first")
            return
        image = Image.open(self.img1path)
        image = transform(image).unsqueeze(0).to(device)
        
        model = VGG16_BN().to(device)
        model.load_state_dict(torch.load("./model/vgg16_mnist.pth", map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            # print the probabilities float of each class

        probabilities = probabilities.cpu().squeeze().numpy()

        # Plot the probabilities for all classes
        plt.figure(figsize=(10, 5))
        plt.bar(range(10), probabilities, color='skyblue')
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.title("Class Probability Distribution")
        plt.xticks(range(10))  # Classes: 0-9
        plt.show()

    def ShowImages(self):
        inference_root = "./Q2_InferenceData"
        inference_dataset = InferenceDataset(inference_root)    
        cat_image, _ = next((img, path) for img, path in inference_dataset if "Cat" in path)
        dog_image, _ = next((img, path) for img, path in inference_dataset if "Dog" in path)

        cat_image = self.denormalize_and_convert(cat_image)
        dog_image = self.denormalize_and_convert(dog_image)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
        axes[0].imshow(cat_image)
        axes[0].set_title("Cat")
        axes[0].axis("off")

        # Display the dog image on the second subplot
        axes[1].imshow(dog_image)
        axes[1].set_title("Dog")
        axes[1].axis("off")

        # Show the combined figure
        plt.show()

    def denormalize_and_convert(self, image_tensor):
        image = image_tensor.permute(1, 2, 0).numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = (image * 255).astype(np.uint8)
        return image




    def ShowStructure02(self):
        model = ResNet50().to(device)
        model.load_state_dict(torch.load("./model/resnet50_cats_dogs.pth", map_location=device, weights_only=True))
        model.eval()
        summary(model, (3, 224, 224))

    def ShowComparison(self):
        img = cv2.imread("Compare_accuracy.png")
        cv2.imshow("Comparison", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Inference(self):
        if self.img2path == "":
            print("Please load an image first")
            return
        image = Image.open(self.img2path)
        image = transform02(image).unsqueeze(0).to(device)

        model = ResNet50().to(device)
        model.load_state_dict(torch.load("./model/resnet50_cats_dogs.pth", map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        labels = {0: "Cat", 1: "Dog"}

        print(f"Prediction: {labels[predicted_class]}")
        self.lblpredict.setText(f"Prediction: {labels[predicted_class]}")

        

    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = UI_MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())




