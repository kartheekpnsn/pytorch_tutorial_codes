import torch

"""
Create a backpropogation for a simple regression
y = w * x
loss = (y - yhat)^2 (Squared error)
"""
# Training data
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# Forward pass and compute loss
yhat = w * x
loss = (y - yhat) ** 2
print(f'Loss is: {loss.item()}')

# Backward pass
loss.backward()
print(f'After 1 iteration - Final Gradient: {w.grad.item()}')

# Update weights
# Re-run forward and backward pass until the loss is minimized
