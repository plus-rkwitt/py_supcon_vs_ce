import torch
import torch.nn as nn


class LinearView(nn.Module):
    """
    Layer which applies a ``view`` operation to its inputs such that all 
    dimensions after the first are flattened, e.g., 

    ::
      l = LinearView()
      x = torch.randn(7, 10, 5)

      y = l(x)
      y.size() # = (7, 50)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class Apply(nn.Module):
    """
    Module wrapper for an arbitrary function.
    """

    def __init__(self, function):
        """[summary]

        Args:

            function ([callable]): wrappee

        Returns:
            Applies ``function`` to input.
        """
        super().__init__()
        self.function = function

    def forward(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class GroupedLinear(nn.Module):
    r"""Appies a group-wise linear operation to the input."""

    def __init__(self, in_features, out_features, groups, bias=True):
        super().__init__()

        self._m = nn.Conv1d(in_features,
                            out_features,
                            kernel_size=1,
                            groups=groups,
                            bias=bias)

        self.out_features = out_features

    def forward(self, x):
        bs = x.size(0)

        x = x.unsqueeze(-1)
        x = self._m(x)
        x = x.view(bs, self.out_features)

        return x


class View(nn.Module):
    """
    Module version of ``torch.view``. 
    """

    def __init__(self, view_args):
        super().__init__()
        self.view_args = view_args

    def forward(self, input):
        return input.view(*self.view_args)
