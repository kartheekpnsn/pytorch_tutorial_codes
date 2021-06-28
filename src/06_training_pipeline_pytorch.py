import numpy as np
import torch
import torch.nn as nn

"""
Inorder to comeup with our own model/custom linear regression model
"""
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(f'Number of Samples: {n_samples}, Number of features: {n_features}')
input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item()}')
## Training ------------------------------------------------------------------------------------------------------------
loss = nn.MSELoss()
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
n_iters = 100
for epoch in range(n_iters):
    yhat = model(X)
    l = loss(Y, yhat)
    l.backward()  # Calculates Gradient: dl/dw
    # Update the weights -----------------------------------------------------------------------------------------------
    optimizer.step()
    # Empty the gradients ----------------------------------------------------------------------------------------------
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(w)
        print(f'Epoch: {epoch + 1}: Weight: {w[0][0].item()}, Loss: {l:.8f}')
print(f'Prediction after training: f(5) = {model(X_test).item()}')
print("==>" * 20)
