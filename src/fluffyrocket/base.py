"""Base class for MiniRocket-like transformers."""

import abc
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
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

    Subclass should implement :meth:`ppv`.

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

    def forward(self, x):
        _, num_channels, _ = x.shape

        features = []
        feature_index_start = 0
        combination_index = 0
        num_channels_start = 0

        for i in range(len(self.dilations)):
            dilation = self.dilations[i].item()
            padding = ((9 - 1) * dilation) // 2
            num_features_this_dilation = self.num_features_per_dilation[i].item()

            C = F.conv1d(
                x, self.kernels, padding=padding, dilation=dilation, groups=num_channels
            )
            C = C.view(-1, num_channels, 84, C.shape[-1])

            for j in range(84):
                feature_index_end = feature_index_start + num_features_this_dilation
                num_channels_this_combination = self.num_channels_per_combination[
                    combination_index
                ].item()
                num_channels_end = num_channels_start + num_channels_this_combination
                channels_this_combination = self.channel_indices[
                    num_channels_start:num_channels_end
                ]

                C_sum = torch.sum(C[:, channels_this_combination, j, :], dim=1)

                biases_this_kernel = self.biases[feature_index_start:feature_index_end]

                if (i + j) % 2 == 0:
                    ppv = self.ppv(C_sum.unsqueeze(-1), biases_this_kernel)
                else:
                    ppv = self.ppv(
                        C_sum[:, padding:-padding].unsqueeze(-1), biases_this_kernel
                    )
                features.append(ppv)

                feature_index_start = feature_index_end
                combination_index += 1
                num_channels_start = num_channels_end

        return torch.cat(features, dim=1)

    @abc.abstractmethod
    def ppv(self, x, biases):
        raise NotImplementedError
