import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# Paths to training and validation datasets
train_dir = "./drive/MyDrive/dataset/training_dataset"
val_dir = "./drive/MyDrive/dataset/validation_dataset"

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 requires 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet50 expects normalized inputs
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Print class-to-index mapping
print("Class to index mapping:", train_dataset.class_to_idx)

# Define the model
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

model = ResNet50().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and validation loop
train_losses, val_losses = [], []
train_accs, val_accs = [], []

def train_and_validate():
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss, correct_train = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            scaler = torch.cuda.amp.GradScaler()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)

        # Validation phase
        model.eval()
        val_loss, correct_val = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)

        # Calculate average loss and accuracy
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_acc = correct_train.double() / len(train_loader.dataset)
        val_acc = correct_val.double() / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Train the model
train_and_validate()

# Save the trained model
torch.save(model.state_dict(), "resnet50_cats_dogs.pth")
print("Model saved to 'resnet50_cats_dogs.pth'")
