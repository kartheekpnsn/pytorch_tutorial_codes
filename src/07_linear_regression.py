"""
1) Design Model (input size, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction and loss
    - backward pass: gradients
    - update weights
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 1) Prepare Data
X_numpy, y_numpy= datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
print(X_numpy.shape, y_numpy.shape)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape

# 2) Prepare Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 3) Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 4) Training loop
n_epochs = 100
for epoch in range(1, n_epochs+1):
    # forward pass and loss
    yhat = model(X)
    loss = criterion(yhat, y)

    # backward pass
    loss.backward()

    # Weight update
    optimizer.step()

    # Empty the gradients
    optimizer.zero_grad()

    # Print information
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}:: Loss = {loss.item():.4f}')

# 5) Plot the values
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'bo')
plt.plot(X_numpy, predicted, 'r')
plt.show()

