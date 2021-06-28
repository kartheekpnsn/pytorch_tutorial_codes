import numpy as np
import torch
import torch.nn as nn

"""
Step 1
1. Prediction: Manually
2. Gradient Computation: Manually
3. Loss Computation: Manually
4. Parameter updates: Manually
"""
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0


def forward(x):
    """
    Forwards Pass
    :param x:
    :return:
    """
    return w * x


def loss(y, yhat):
    """
    Calculate Loss: MSE
    :param y:
    :param yhat:
    :return:
    """
    return ((y - yhat) ** 2).mean()


def gradient(x, y, yhat):
    """
    Calculate gradient w.r.t w
    - MSE (J) = 1/N * (w*x - y)**2
    - dJ/dw = 1/N 2x (w*x - y)
    :param x:
    :param y:
    :param yhat:
    :return:
    """
    return np.dot(2 * x, yhat - y).mean()


print(f'Prediction before training: f(5) = {forward(5)}')
## Training ------------------------------------------------------------------------------------------------------------
lr = 0.01
n_iters = 20
for epoch in range(n_iters):
    yhat = forward(X)
    l = loss(Y, yhat)
    dw = gradient(X, Y, yhat)
    # Update the weights -----------------------------------------------------------------------------------------------
    w = w - (lr * dw)
    if epoch % 2 == 0:
        print(f'Epoch: {epoch + 1}: Weight: {w}, Loss: {l:.8f}')

print(f'Prediction before training: f(5) = {forward(5):.5f}')
print("==>" * 20)
"""
Step 2
1. Prediction: Manually
2. Gradient Computation: AutoGrad
3. Loss Computation: Manually
4. Parameter updates: Manually
"""
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
print(f'Prediction before training: f(5) = {forward(5)}')
## Training ------------------------------------------------------------------------------------------------------------
lr = 0.01
n_iters = 100
for epoch in range(n_iters):
    yhat = forward(X)
    l = loss(Y, yhat)
    l.backward()  # Calculates Gradient: dl/dw
    # Update the weights -----------------------------------------------------------------------------------------------
    with torch.no_grad():
        w -= (lr * w.grad)
    # Empty the gradients ----------------------------------------------------------------------------------------------
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch + 1}: Weight: {w}, Loss: {l:.8f}')
print(f'Prediction before training: f(5) = {forward(5):.5f}')
print("==>" * 20)
"""
Step 3
1. Prediction: Manually
2. Gradient Computation: AutoGrad
3. Loss Computation: PyTorch Loss
4. Parameter updates: PyTorch Optimizers
"""
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
print(f'Prediction before training: f(5) = {forward(5)}')
# 1) Design model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update the weights
loss = nn.MSELoss()
lr = 0.01
optimizer = torch.optim.SGD([w], lr=lr)
print(f'Prediction before training: f(5) = {forward(5)}')
## Training ------------------------------------------------------------------------------------------------------------
n_iters = 100
for epoch in range(n_iters):
    yhat = forward(X)
    l = loss(Y, yhat)
    l.backward()  # Calculates Gradient: dl/dw
    # Update the weights -----------------------------------------------------------------------------------------------
    optimizer.step()
    # Empty the gradients ----------------------------------------------------------------------------------------------
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch + 1}: Weight: {w}, Loss: {l:.8f}')
print(f'Prediction before training: f(5) = {forward(5):.5f}')
print("==>" * 20)
"""
Step 4
1. Prediction: PyTorch Model
2. Gradient Computation: AutoGrad
3. Loss Computation: PyTorch Loss
4. Parameter updates: PyTorch Optimizers
"""
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(f'Number of Samples: {n_samples}, Number of features: {n_features}')
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)
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
