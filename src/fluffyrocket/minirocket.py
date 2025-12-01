"""PyTorch implementation of the original MiniRocket with hard PPV."""

from .base import MiniRocketBase

__all__ = [
    "MiniRocket",
]


class MiniRocket(MiniRocketBase):
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
    >>> torch.allclose(torch.from_numpy(trf_original), trf_torch)
    True
    """

    def ppv(self, x, biases):
        return (x > biases).float().mean(1)
