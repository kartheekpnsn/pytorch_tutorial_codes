"""
Activation Functions:
- Non linear transformation to decide whether a neuron should be activated or not
- Without activation functions, it is basically stacked linear regressions
- Hence we need non-linear to learn more complex tasks
- After each layer, we use an activation functions

Popular ones:
- Step function
    - if x>= threshold, 1, 0
    - Not used in practice. Instead use Sigmoid
- Sigmoid (nn.Sigmoid or torch.sigmoid)
    - 1/(1+exp(-x)): outputs a probability b/w [0, 1]
    - Typically last layer of binary classification problem
- TanH (nn.TanH or torch.tanh)
    - (2/(1 + exp(-2x)) - 1
    - Scaled and Shifted Sigmoid
    - Returns in range [-1, 1]
    - Typically used after hidden layers
- ReLU (nn.ReLU or torch.relu)
    - Rectified Linear Unit
    - Most popular choice (if you don't know, use ReLU)
    - max(0, x): Returns 0 for -ve values, x for +ve values
- Leaky ReLU (nn.LeakyReLU or F.leaky_relu())
    - Improved version of ReLU
    - Tries to solve vanishing gradient problem
    - if x>=0, x, a*x (a=typically very small, 0.0001)
- Softmax (torch.softmax)
    - exp(x)/sum(exp(x))
    - Squash the inputs to probability scores
    - Typically used as last layer of multi-class classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


class NeuralNet2(nn.Module):
    """
    Use activation functions directly in forward pass
    """

    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        # F.leaky_relu()
        return out
