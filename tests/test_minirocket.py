import torch

from fluffyrocket import FluffyRocketFeatures


def _feature_extractor(sharpness):
    return FluffyRocketFeatures(
        c_in=1,
        seq_len=9,
        num_features=84,
        random_state=0,
        sharpness=sharpness,
    )


def test_none_sharpness_uses_hard_ppv():
    model = _feature_extractor(sharpness=None)
    convolutions = torch.tensor([[[0.0, 1.0, 2.0]]])
    biases = torch.tensor([[0.5, 1.5]])

    result = model._get_PPVs(convolutions, biases)

    expected = torch.tensor([[2 / 3, 1 / 3]])
    torch.testing.assert_close(result, expected)


def test_finite_sharpness_uses_soft_ppv():
    sharpness = 2.0
    model = _feature_extractor(sharpness=sharpness)
    convolutions = torch.tensor(
        [[[0.0, 1.0, 2.0]], [[-1.0, 0.0, 1.0]]], requires_grad=True
    )
    biases = torch.tensor([[0.5, 1.5]])

    result = model._get_PPVs(convolutions, biases)

    expected = (
        torch.sigmoid(
            sharpness * (convolutions.detach().unsqueeze(-1) - biases.view(1, 1, 1, 2))
        )
        .mean(dim=2)
        .flatten(1)
    )
    torch.testing.assert_close(result, expected)
    assert result.shape == (2, 2)

    result.sum().backward()
    assert convolutions.grad is not None
    assert torch.all(convolutions.grad > 0)
