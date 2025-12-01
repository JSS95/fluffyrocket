"""Base class for MiniRocket-like transformers."""

import abc
from itertools import combinations

import numpy as np
import torch
from torch.nn import Module

from ._minirocket import fit as _fit

_INDICES = np.array(list(combinations(np.arange(9), 3)), dtype=np.int32)
_KERNELS = np.full((84, 1, 9), -1.0, dtype=np.float32)
_KERNELS[np.arange(84)[:, np.newaxis], 0, _INDICES] = 2.0

__all__ = [
    "MiniRocketBase",
]


class MiniRocketBase(Module, abc.ABC):
    """Base class for MiniRocket-like transformers.

    Parameters
    ----------
    num_features : int, default=10,000
    max_dilations_per_kernel : int, default=32
    random_state : int, default=None
    """

    def __init__(
        self,
        num_features=10_000,
        max_dilations_per_kernel=32,
        random_state=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit dilation and biases.

        Parameters
        ----------
        X : array of shape ``(num_examples, num_channels, input_length)``.
            Data type must be float32.
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        _, num_channels, _ = X.shape
        kernels = torch.from_numpy(_KERNELS).repeat(num_channels, 1, 1)
        self.register_buffer("kernels", kernels)

        (
            num_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            biases,
        ) = _fit(X, self.num_features, self.max_dilations_per_kernel, self.random_state)

        self.register_buffer(
            "num_channels_per_combination",
            torch.from_numpy(num_channels_per_combination),
        )
        self.register_buffer("channel_indices", torch.from_numpy(channel_indices))
        self.register_buffer("dilations", torch.from_numpy(dilations))
        self.register_buffer(
            "num_features_per_dilation",
            torch.from_numpy(num_features_per_dilation),
        )
        self.register_buffer("biases", torch.from_numpy(biases))

        return self

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError
