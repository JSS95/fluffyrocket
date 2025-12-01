"""PyTorch implementation of MiniRocket with soft PPV."""

import torch

from .base import MiniRocketBase

__all__ = [
    "FluffyRocket",
]


class FluffyRocket(MiniRocketBase):
    """PyTorch MiniRocket with soft PPV.

    Parameters
    ----------
    sharpness : float, default=10.0
        Sharpness parameter for the sigmoid function used to compute soft PPV.
    num_features : int, default=10,000
    max_dilations_per_kernel : int, default=32
    learnable : bool, default=False
        Whether the sharpness parameter is learnable.
    random_state : int, default=None

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> import torch
    >>> from fluffyrocket import FluffyRocket
    >>> from fluffyrocket._minirocket import fit, transform
    >>> X, _ = load_unit_test()
    >>> X = X.astype("float32")
    >>> trf_original = transform(X, fit(X, num_features=84, seed=42))

    Small sharpness gives a smoother approximation of PPV, making it more
    differentiable.

    >>> fluffyrocket = FluffyRocket(1.0, num_features=84, random_state=42).fit(X)
    >>> trf_torch = fluffyrocket(torch.from_numpy(X))
    >>> torch.allclose(torch.from_numpy(trf_original), trf_torch)
    False

    Large sharpness approximates hard PPV used in the original MiniRocket.

    >>> fluffyrocket = FluffyRocket(1000, num_features=84, random_state=42).fit(X)
    >>> trf_torch = fluffyrocket(torch.from_numpy(X))
    >>> torch.allclose(torch.from_numpy(trf_original), trf_torch)
    True
    """

    def __init__(
        self,
        sharpness=10.0,
        num_features=10_000,
        max_dilations_per_kernel=32,
        learnable=False,
        random_state=None,
    ):
        super().__init__(num_features, max_dilations_per_kernel, random_state)
        assert sharpness > 0, "Sharpness must be positive."
        log_sharpness = torch.log(torch.tensor(sharpness))  # Ensure positivity
        if learnable:
            self.log_sharpness = torch.nn.Parameter(log_sharpness)
        else:
            self.register_buffer("log_sharpness", log_sharpness)

    @property
    def sharpness(self):
        return torch.exp(self.log_sharpness)

    def ppv(self, x, biases):
        return torch.sigmoid(self.sharpness * (x - biases)).mean(1)
