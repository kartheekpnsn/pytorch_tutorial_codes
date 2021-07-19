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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#  Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
n_epochs = 4
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


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get some random data
dataiter = iter(train_loader)
images, labels = dataiter.next()

# Show the sample image
imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2) # Kernel = 2, Stride = 2
conv2 = nn.Conv2d(6, 16, 5)
print(images.shape)  # 4, 3, 32, 32
x = conv1(images)
# (((InputSize - FilterSize) + 2*PaddingSize)/StrideSize) + 1
# (((32 - 5) + 2*0)/1) + 1
print(x.shape)  # 4, 6, 28, 28
x = pool(x)
print(x.shape)  # 4, 6, 14, 14
x = conv2(x)
# (((InputSize - FilterSize) + 2*PaddingSize)/StrideSize) + 1
# (((14 - 5) + 2*0)/1) + 1
print(x.shape)  # 4, 6, 10, 10
x = pool(x)
print(x.shape)  # 4, 16, 5, 5