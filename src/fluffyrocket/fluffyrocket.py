"""PyTorch implementation of MiniRocket with soft PPV."""

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
    random_state : int, default=None
    """

    def __init__(
        self,
        sharpness=10.0,
        num_features=10_000,
        max_dilations_per_kernel=32,
        random_state=None,
    ):
        super().__init__(num_features, max_dilations_per_kernel, random_state)
        self.sharpness = sharpness
