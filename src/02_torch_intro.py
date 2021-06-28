import numpy as np
import torch

print(f"Torch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

# Basic tensor creations
print("==> Basic tensor creations")
print(torch.empty(3, 3, 3))
print(torch.rand(3))
print(torch.rand(2, 2))
print(torch.zeros(2, 2))
print(torch.ones(3, 3, 3))

# Create tensor with datatype
print("==> Create tensor with datatype")
x = torch.ones(3, 3, dtype=torch.int)  # torch.double, #torch.float16
print(x.dtype)
print(x.size())

# Create tensor from list
print("==> Create tensor from list")
x = torch.tensor([1, 2, 3])

# Basic operations
print("==> Basic operations")
x = torch.rand(2, 2)
y = torch.rand(2, 2)
z = x + y  # Element wise additions
z = torch.add(x, y)  # torch.sub(x, y) # torch.mul(x, y) # torch.div(x, y)
y.add_(x)  # Modify y (inplace: Where the function that has trailing "_")
print(z)

# Slicing operations
print("==> Slicing operations")
x = torch.rand(5, 3)
print(x[:, 0])  # First column and all rows
print(x[1, :])  # Second row and all columns
print(x[1, 1])  # Returns tensor at 2nd row, 2nd column
print(x[1, 1].item())  # Returns value (instead of tensor)

# Reshaping tensors
print("==> Reshaping tensors")
x = torch.rand(4, 4)
y = x.view(16)  # into 1D tensor
print(y)
y = x.view(-1, 8)  # into 2D tensor (2, 8)
print(y)
print(y.size())

# Torch -> Numpy
print("==> Torch -> Numpy")
a = torch.ones(5)
b = a.numpy()
print(type(b))
a.add_(1)  # Adds 1 inplace, also replace b
print(a)
print(b)

# Numpy -> Torch
print("==> Numpy -> Torch")
a = np.ones(5)
b = torch.from_numpy(a)
a = a + 1  # Adds 1, also replace b
print(a)
print(b)

# Below code uses GPU
print("==> Below code uses GPU")
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)  # moves to cuda device (GPU)
    z = x + y  # This happens on GPU
    print(z.numpy())  # Returns error as numpy cannot handle GPU tensors
    z = z.to("cpu")  # Move the tensor back to CPU so that it can be used in numpy
    print(z.numpy())

# How to specify torch that we need to calculate gradients on a tensor
print("==> How to specify torch that we need to calculate gradients on a tensor")
x = torch.ones(5, requires_grad=True)
print(x)