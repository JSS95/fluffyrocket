"""PyTorch implementation of the original MiniRocket with hard PPV."""

from torch.nn import Module

from ._minirocket import fit as _fit

__all__ = [
    "MiniRocket",
]


class MiniRocket(Module):
    """PyTorch MiniRocket [1]_ with hard PPV.

    This class aims to exactly reproduce transformation result from
    the original MiniRocket model.

    Parameters
    ----------
    num_features : int, default=10,000
    max_dilations_per_kernel : int, default=32
    random_state : int, default=None

    References
    ----------
    .. [1] Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
       "Minirocket: A very fast (almost) deterministic transform for
       time series classification." Proceedings of the 27th ACM SIGKDD
       conference on knowledge discovery & data mining. 2021.

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> import torch
    >>> from fluffyrocket import MiniRocket
    >>> from fluffyrocket._minirocket import fit, transform
    >>> X, _ = load_unit_test()
    >>> X = X.astype("float32")
    >>> trf_original = transform(X, fit(X, num_features=84, seed=42))
    >>> minirocket = MiniRocket(num_features=84, random_state=42).fit(X)
    >>> trf_torch = minirocket(torch.from_numpy(X))
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
        (
            self.num_channels_per_combination,
            self.channel_indices,
            self.dilations,
            self.num_features_per_dilation,
            self.biases,
        ) = _fit(X, self.num_features, self.max_dilations_per_kernel, self.random_state)
        return self

    def forward(self, x):
        pass
