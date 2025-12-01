"""PyTorch implementation of the original MiniRocket with hard PPV."""

from torch.nn import Module

__all__ = ["MiniRocket"]


class MiniRocket(Module):
    """PyTorch MiniRocket with hard PPV.

    This class aims to exactly reproduce transformation result from
    the original MiniRocket model.

    Parameters
    ----------

    References
    ----------
    .. [1] Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
       "Minirocket: A very fast (almost) deterministic transform for
       time series classification." Proceedings of the 27th ACM SIGKDD
       conference on knowledge discovery & data mining. 2021.
    """
