"""
epoch = 1: 1 complete forward and backward pass for ALL training samples
batch_size = number of training samples in 1 forward and backward pass
n_iterations = number of passes, each pass using [batch_size] number of samples

e.g: 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch.
"""
import pandas as pd
import torch
import torchvision
import math
from torch.utils.data import DataLoader, Dataset


class WineDataSet(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = pd.read_csv("../data/wine.csv")
        independent_cols = [c for c in xy.columns if c != 'Wine']
        target = 'Wine'
        # Note: We don't convert the values to tensors
        self.x = xy[independent_cols].values
        self.y = xy[[target]].values  # Shape: n_samples, 1
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # gets number of samples
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

dataset = WineDataSet(transform=ToTensor())
inputs, targets = dataset[0]
print(type(inputs), type(targets))

dataset = WineDataSet(transform=None)
inputs, targets = dataset[0]
print(inputs)
print(type(inputs), type(targets))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataSet(transform=composed)
inputs, targets = dataset[0]
print(type(inputs), type(targets))
print(inputs)