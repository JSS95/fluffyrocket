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
        random_state=None,
    ):
        super().__init__(num_features, max_dilations_per_kernel, random_state)
        self.sharpness = sharpness

    def forward(self, x):
        import torch
        import torch.nn.functional as F

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
                    ppv = torch.sigmoid(
                        self.sharpness * (C_sum.unsqueeze(-1) - biases_this_kernel)
                    ).mean(1)
                else:
                    ppv = torch.sigmoid(
                        self.sharpness
                        * (
                            C_sum[:, padding:-padding].unsqueeze(-1)
                            - biases_this_kernel
                        )
                    ).mean(1)
                features.append(ppv)

                feature_index_start = feature_index_end
                combination_index += 1
                num_channels_start = num_channels_end

        return torch.cat(features, dim=1)
