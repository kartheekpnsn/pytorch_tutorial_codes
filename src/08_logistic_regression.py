"""
1) Design Model (input size, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction and loss
    - backward pass: gradients
    - update weights
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 0) Prepare Data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
print(n_samples, n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1) Design Model: f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, X):
        yhat = torch.sigmoid(self.linear(X))
        return yhat


model = LogisticRegression(input_size=n_features)

# 2) Loss and Optimizer
learning_rate = 0.01
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
n_epochs = 200
for epoch in range(1, n_epochs):
    # forward pass and loss
    yhat = model(X_train)
    loss = criterion(yhat, y_train)

    # backward pass
    loss.backward()

    # weight updates
    optimizer.step()

    # empty the gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}:: Loss = {loss.item():.4f}')

# Prediction on test set
with torch.no_grad():
    yhat = model(X_test).detach().numpy()
    # predicted = np.where(predicted.flatten() >= 0.5, 1, 0)
    yhat = yhat.round()
    print(f'Accuracy: {accuracy_score(y_test, yhat):.4f}')
