"""Minirocket with soft PPV."""

import torch
from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures

__all__ = [
    "FluffyRocketFeatures",
]


class FluffyRocketFeatures(MiniRocketFeatures):
    """MiniRocket with soft PPV.

    The *sharpness* parameter controls the sharpness of the sigmoid function
    used to compute the soft PPV. ``None`` indicates infinite sharpness, i.e.,
    the original hard PPV.
    """

    def __init__(
        self,
        c_in,
        seq_len,
        num_features=10_000,
        max_dilations_per_kernel=32,
        random_state=None,
        sharpness=None,
    ):
        super().__init__(
            c_in, seq_len, num_features, max_dilations_per_kernel, random_state
        )
        self.sharpness = sharpness

    def _get_PPVs(self, C, bias):
        """Return hard or differentiable soft proportions of positive values."""
        if self.sharpness is None:
            return super()._get_PPVs(C, bias)

        C = C.unsqueeze(-1)
        bias = bias.view(1, bias.shape[0], 1, bias.shape[1])
        return torch.sigmoid(self.sharpness * (C - bias)).mean(2).flatten(1)
