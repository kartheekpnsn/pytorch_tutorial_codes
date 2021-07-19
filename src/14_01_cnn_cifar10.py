"""
CNN:
- Similar to NN
- Mainly used on Images
- Has convolutional layers + Activation + Pooling layers
    - Learns features from images
- Convolution = Multiplying and summing up the 2 matrices
- Filter applied on the image, then multiplied and summed up
- Pooling:
    - Reduces computation
    - Reduces over fitting (by providing abstractive information of the image)
- Max pooling used to learn edges
- Mean pooling used to smoothen the image
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#  Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
n_epochs = 2
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range (0, 1)
# We transform them to tensors of normalized range (-1, 1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 data
train_dataset = torchvision.datasets.CIFAR10(root="../cifar10_data", train=True, transform=transform,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root="../cifar10_data", train=False, transform=transform,
                                            download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Implement ConvNet
class ConvNet(nn.Module):
    """
    1) Conv Layer + ReLU
    2) Max Pooling
    3) Conv Layer + ReLU
    4) Max Pooling
    5) 3 Different Fully Connected layers
    6) Softmax & Cross entropy (don't need to implement this)
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5)  # output size = ((32-5) + 0)/1 + 1 = 28 x 28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # output size = 14 x 14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5)  # output size = ((14-5)+0)/1 + 1 = 10 x 10
        self.fc1 = nn.Linear(in_features=16 * 5 * 5,
                             out_features=120)  # As we apply one more padding 5 x 5 and 16 channels
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Before passing to Fully connected, flatten it
        x = x.view(-1, 16 *5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Note: no softmax, it is included in loss
        return x


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(1, n_epochs + 1):
    for i, (images, labels) in enumerate(train_loader):
        # Current shape: [4, 3, 32, 32] = 4, 3, 1024
        # input layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()  # empty the gradients
        loss.backward()
        optimizer.step()  # updates the parameters

        if (i + 1) % 100 == 0:
            print(f'Epoch: {epoch}/{n_epochs}, step: {i + 1}/{n_total_steps}, loss={loss.item():.3f}')

print("<== Training Finished ==>")
# Prediction on test data
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    accuracy = 100 * (n_correct / n_samples)
    print(f'Accuracy on test set: {accuracy} %..')
    for i in range(10):
        acc = 100 * (n_class_correct[i] / n_class_samples[i])
        print(f'Accuracy of {classes[i]}: {acc} %..')
