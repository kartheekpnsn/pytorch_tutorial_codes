"""
1) MNIST
2) DataLoader, Transformation
3) Multilayer NN, activation function
4) Loss and optimizer
5) Training loop (batch training)
6) Model evaluation
7) GPU Support (optional)
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Initialize a writer for tensor board
writer = SummaryWriter("../runs/mnist/")
# In CLI: tensorboard --logdir=runs

#  Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784  # 28x28
hidden_size = 100
output_size = 10  # 10 classes in MNIST
n_epochs = 2
batch_size = 100
learning_rate = 0.001

# Load MNIST data
train_dataset = torchvision.datasets.MNIST(root="../mnist_data", train=True, transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root="../mnist_data", train=False, transform=transforms.ToTensor(),
                                          download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load Some samples from data loader
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

# Plot the samples
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')
# Add images to tensorboard
image_grid = torchvision.utils.make_grid(samples)
writer.add_image("mnist_images", image_grid)
writer.close()


# Feed forward Neural Netword
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        # Note: no softmax, it is included in loss
        return out


model = NeuralNet(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1, 28 * 28))

# Training loop
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0.0
for epoch in range(1, n_epochs + 1):
    for i, (images, labels) in enumerate(train_loader):
        # Current shape: 100, 1, 28, 28
        # Reshape to: 100, 784
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()  # empty the gradients
        loss.backward()
        optimizer.step()  # updates the parameters

        running_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        running_correct += (predictions == labels).sum().item()
        if (i + 1) % 100 == 0:
            print(f'Epoch: {epoch}/{n_epochs}, step: {i + 1}/{n_total_steps}, loss={loss.item():.3f}')
            writer.add_scalar("training_loss", running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar("accuracy", running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0.0

# Prediction on test data
label_list = []
pred_list = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        label_list.append(predictions)
        pred_list.append(class_predictions)

    label_list = torch.cat(label_list)
    pred_list = torch.cat([torch.stack(batch) for batch in pred_list])

    accuracy = 100 * (n_correct / n_samples)
    print(f'Accuracy on test set: {accuracy} %..')

    classes = 10
    for i in range(classes):
        labels_i = label_list == i
        preds_i = pred_list[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
