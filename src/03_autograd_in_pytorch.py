import torch

"""
AutoGrad:
---------
Automatic Differentiation is a building block of not only PyTorch, but every DL library out there. 
In my opinion, PyTorch's automatic differentiation engine, called Autograd is a brilliant tool to understand 
how automatic differentiation works. This will not only help you understand PyTorch better, 
but also other DL libraries.

Aim is to create gradients of some function w.r.t 'x'
- use requires_grad = True
- When we do some operations with this tensor, pytorch will create a computational graph for us
"""

x = torch.randn(3, requires_grad=True)
y = x + 2  # This creates a computational graph, graph_fn attribute is added
print(y)
z = y * y * 2
print(z)
z = z.mean()
print(z)
z.backward()  # dz/dx
print(x.grad)  # uses chain rule / vector jacobian rule

"""
To prevent torch from tracking the history/gradients/graph_fn (Sometimes we need in training process). 
We can do it in 3 ways:
1) x.requires_grad_(False)
2) y = x.detach()
3) with torch.no_grad():
    y = x + 2
    print(y)
"""

"""
Simulate a dummy model for some example
"""
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(f"==> Iteration: {epoch}, Gradients:")
    print(weights.grad)
    # To do the next iteration, we must empty the gradients
    weights.grad.zero_()

"""
In next script, we will work with pytorch built-in optimizers
"""
weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.sgd(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()
