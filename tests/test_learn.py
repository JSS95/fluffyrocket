import torch
from aeon.datasets import load_unit_test
from sklearn.preprocessing import LabelEncoder

from fluffyrocket import FluffyRocket


def test_sharpness_learn():
    """Test if sharpness parameter is learnable."""
    # Load classification dataset.
    X, y = load_unit_test()
    X = X.astype("float32")

    # Not-trainable model
    fluffy = FluffyRocket(1.0, num_features=84, learnable=False, random_state=42)
    fluffy.fit(X)
    model = torch.nn.Sequential(
        fluffy,
        torch.nn.Linear(84, 2),
    )
    initial_sharpness = model[0].sharpness.clone()
    trained_sharpness = train_by_one_step(model, X, y)[0].sharpness.clone()
    assert initial_sharpness == trained_sharpness

    # Trainable model
    fluffy = FluffyRocket(1.0, num_features=84, learnable=True, random_state=42)
    fluffy.fit(X)
    model = torch.nn.Sequential(
        fluffy,
        torch.nn.Linear(84, 2),
    )
    initial_sharpness = model[0].sharpness.clone()
    trained_sharpness = train_by_one_step(model, X, y)[0].sharpness.clone()
    assert initial_sharpness != trained_sharpness


def train_by_one_step(model, X, y):
    X = torch.from_numpy(X)
    y = torch.from_numpy(LabelEncoder().fit_transform(y))

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    return model
