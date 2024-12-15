import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

# Load data and preprocess
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)   

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# define VGG16 model
class VGG16_BN(nn.Module):
    def __init__(self, num_classes = 10):
        super(VGG16_BN, self).__init__()
        self.vgg16 = models.vgg16_bn(pretrained=False)
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16(x)

model = VGG16_BN().to(device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train and validate
train_losses, val_losses = [], []
train_accs, val_accs = [], []

def train_and_validate():
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct_train = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_train += torch.sum(predicted == labels.data)

        model.eval()
        val_loss, correct_val = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_val += torch.sum(predicted == labels.data)

        # calculate average loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_acc = correct_train / len(train_loader.dataset)
        val_acc = correct_val / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# start training
train_and_validate()

# draw training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(EPOCHS), train_losses, label='Training Loss')
plt.plot(np.arange(EPOCHS), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.arange(EPOCHS), train_accs, label='Training Accuracy')
plt.plot(np.arange(EPOCHS), val_accs, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# save the figure
plt.savefig('training_results.png')
plt.show()

# save the model and results
torch.save(model.state_dict(), 'vgg16_mnist.pth')
print("Model and results saved!")

        