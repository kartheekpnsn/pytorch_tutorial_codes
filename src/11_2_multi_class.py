import torch.nn as nn


class NeuralNet2(nn.Module):

    def __init__(self, input_size, hidden_size, n_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax needed
        return out


model = NeuralNet2(input_size=28 * 28, hidden_size=5, n_classes=3)
criterion = nn.CrossEntropyLoss()  # (Also applies softmax)
