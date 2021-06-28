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

    def __init__(self):
        # data loading
        xy = pd.read_csv("../data/wine.csv")
        independent_cols = [c for c in xy.columns if c != 'Wine']
        target = 'Wine'
        self.x = torch.from_numpy(xy[independent_cols].values)
        self.y = torch.from_numpy(xy[[target]].values)  # Shape: n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # gets number of samples
        return self.n_samples


# Using Dataset
dataset = WineDataSet()
features, labels = dataset[0]

# Using Data Loader
batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)

# Dummy training loop
n_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)
print(total_samples, n_iterations)

for epoch in range(1, n_epochs+1):
    for idx, (inputs, labels) in enumerate(dataloader):
        # forward pass
        # backward pass
        # update weights
        if (idx + 1) % 5 == 0:
            print(f'Epoch: {epoch}/{n_epochs}, step: {idx+1}/{n_iterations}, inputs: {inputs.shape}')
