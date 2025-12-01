"""MiniRocket with ridge classifier example.

This example is for demonstrating that FluffyRocket can be updated end-to-end
with a ridge classifier using backpropagation, not for performance.

This does not support mini-batch training because the closed-form solution of ridge
classifier requires all data at once. As recommended by the original paper,
use trainable classifiers for practical applications.
"""

import numpy as np
import torch
from aeon.datasets import load_unit_test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.nn import Module, Sequential

from fluffyrocket import FluffyRocket

EPOCH = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RidgeClassifier(Module):
    """Ridge Classifier using closed-form solution for backpropagation."""

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def fit(self, X, y):
        _, n_features = X.shape
        eye = np.eye(n_features, dtype=X.dtype)
        self.register_buffer("eye", torch.from_numpy(eye))
        self.register_buffer("y", torch.from_numpy(y).float())

    def forward(self, x):
        if self.training:
            A = x.T @ x + self.alpha * self.eye
            B = x.T @ self.y
            W = torch.linalg.solve(A, B)
            self.W = W.detach()
            return x @ W
        return x @ self.W


X, y = load_unit_test("train")
X = X.astype("float32")
y = LabelBinarizer(pos_label=1, neg_label=-1).fit_transform(y).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

fluffy = FluffyRocket(1.0, num_features=100, learnable=True, random_state=42)
fluffy.fit(X_train)
ridge = RidgeClassifier(alpha=1.0)
ridge.fit(fluffy(torch.from_numpy(X_train)).detach().numpy(), y_train)
model = Sequential(fluffy, ridge).to(device)

X_train, X_test, y_train, y_test = map(
    torch.from_numpy, [X_train, X_test, y_train, y_test]
)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

for i in range(EPOCH):
    model.train()
    X_train, y_train = X_train.to(device), y_train.to(device)
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = loss_fn(test_output, y_test)
    print(
        f"Epoch {i+1}, Sharpness: {model[0].sharpness},  Test loss: {test_loss.item()}"
    )
