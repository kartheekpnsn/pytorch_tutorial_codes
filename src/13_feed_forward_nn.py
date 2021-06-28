"""
1) MNIST
2) DataLoader, Transformation
3) Multilayer NN, activation function
4) Loss and optimizer
5) Training loop (batch training)
6) Model evaluation
7) GPU Support (optional)
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()

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
        # Note: no softmax
        return out


model = NeuralNet(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
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

        if (i + 1) % 100 == 0:
            print(f'Epoch: {epoch}/{n_epochs}, step: {i + 1}/{n_total_steps}, loss={loss.item():.3f}')

# Prediction on test data
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    accuracy = 100 * (n_correct/n_samples)
    print(f'Accuracy on test set: {accuracy} %..')