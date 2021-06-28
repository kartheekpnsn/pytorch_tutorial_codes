"""
Softmax: Function that converts a vector of values into 0-1 (and sum of them becomes 1)
- exp(x)/sum(exp(x))
- i.e. Function to convert numbers into probability values (for multi class classification)
- Usually used as final layer of NN

Cross-Entropy: Loss function
- Measures the performance of classification model (where output: [0, 1])
- Loss increases as the predicted probability diverges from actual label
- Used in multi-class classification
- Lower the better (because it is a loss function!)
- D(Yhat, Y) = (-1/N) * SUM(Y_i * log(Yhat_i)
- Note: Y must be a one-hot encoded class. if Y has labels (A, B, C) and Y_i = A, then Y_i = [1, 0, 0]

Sigmoid:
- Binary classification
- In Pytorch nn.BCELoss()
- We need to use sigmoid at the end
"""

import numpy as np
import torch
import torch.nn as nn


def softmax_np(x):
    return np.exp(x) / (np.sum(np.exp(x), axis=0))


x = np.array([2.0, 1.0, 0.1])
outputs = softmax_np(x)
print('Softmax (numpy):', outputs)

x = torch.from_numpy(x)
outputs = torch.softmax(x, dim=0)  # dim=0: computes along first axis
print('Softmax (torch):', outputs)

print("==>" * 20)


def cross_entropy_np(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss / float(predicted.shape[0])


# y must be one-hot encoded
# if class 0: [1, 0, 0]
# if class 1: [0, 1, 0]
# if class 2: [0, 0, 1]
Y = np.array([1, 0, 0])
Yhat_good = np.array([0.7, 0.2, 0.1])
Yhat_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy_np(Y, Yhat_good)
l2 = cross_entropy_np(Y, Yhat_bad)
print(f'Loss (Good Prediction) numpy: {l1}')
print(f'Loss (Bad Prediction) numpy: {l2}')

"""
CAREFUL !!!
1) In torch, the cross entropy loss already applies Softmax + Negative Log Likelihood loss
    - Hence, no Softmax in last layer !
2) Also, Y has class labels, NOT ONE-HOT ENCODED!!!
    - Yhat has raw scores (logits), NO SOFTMAX!!!
"""
loss = nn.CrossEntropyLoss()
# For 3 Samples
Y = torch.tensor([2, 0, 1])
Yhat_good = torch.tensor([[2.0, 1.0, 3.1], [2.0, 1.0, 0.1], [2.0, 4.0, 0.1]])  # nsamples x nclasses
Yhat_bad = torch.tensor([[2.5, 2.0, 0.3], [0.5, 2.0, 0.3], [0.5, 2.0, 3.3]])  # nsamples x nclasses
l1 = loss(Yhat_good, Y)
l2 = loss(Yhat_bad, Y)
print(f'Loss (Good Prediction) numpy: {l1.item()}')
print(f'Loss (Bad Prediction) numpy: {l2.item()}')

# To get predictions from logits !!
_, prediction1 = torch.max(Yhat_good, 1)  # Along first dimensions
_, prediction2 = torch.max(Yhat_bad, 1)  # Along first dimensions
print(prediction1)
print(prediction2)
