import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np


# class SphericalLinear(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, **kwargs):
#         super().__init__()
#         self.w = torch.nn.Parameter(torch.randn(output_dim, input_dim-1))

#     def forward(self, x):

#         w_cos = torch.cos(self.w)
#         w_sin = torch.sin(self.w)

#         ones = torch.ones(self.w.size(
#             0), 1, dtype=self.w.dtype, device=self.w.device)

#         w_cos = torch.cat((w_cos, ones), dim=1)
#         w_sin = torch.cumprod(w_sin, dim=1)
#         w_sin = torch.cat((ones, w_sin), dim=1)

#         w = w_sin * w_cos

#         return torch.matmul(x, w.T)

#     @property
#     def weight(self):
#         return self.w


class FixedSphericalSimplexLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, radius=1.):
        super().__init__()
        self.radius = radius

        def normalize(x): 
            x = x / torch.norm(x, dim=1, p=2, keepdim=True)
            x = x*self.radius
            return x

        points = torch.randn(200, 512)
        points = normalize(points)
        points = torch.nn.Parameter(points)

        opt = torch.optim.SGD([points], lr=0.01)

        for _ in range(1000):

            l = (points.unsqueeze(1) - points.unsqueeze(0)).norm(p=2, dim=-1)
            l = torch.triu(l, diagonal=1)
            l = (1 / l).sum()
    
            l.backward()
            
            opt.step()
    
            points.data = normalize(points.data)

        self.register_buffer('weight', points.data)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)
        

class NormedLinear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 learn_radius=False,
                 radius_init=1.0, **kwargs):
        super().__init__()
        assert isinstance(radius_init, float)
        assert isinstance(learn_radius, bool)

        self._r = torch.tensor(radius_init)

        if learn_radius:
            self._r = torch.nn.Parameter(self._r)

        self._w = torch.nn.Parameter(
            torch.randn(out_features, in_features)
        )

    def forward(self, x):

        w = self._w/ self._w.norm(p=2, dim=1, keepdim=True)
        w = self._w * self._r.abs()

        return torch.nn.functional.linear(x, w)

    @property
    def weight(self):
        return self._w

class SphereProjection(nn.Module):
    def __init__(self, p, learn_radius=False, radius_init=1.0):
        super().__init__()
        assert isinstance(radius_init, float)
        assert isinstance(learn_radius, bool)

        assert isinstance(p, int) and p >= 1
        self.p = p

        self._r = torch.tensor(radius_init)

        if learn_radius:
            self._r = torch.nn.Parameter(self._r)

    def forward(self, x):

        assert x.ndim == 2
        return self._r.abs() * x / torch.norm(x, p=self.p, keepdim=True, dim=1)


class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)


class RowNormalizedLinear(nn.Linear):
    def forward(self, x):
        self.weight.data /= self.weight.data.pow(2).sum(1).sqrt().unsqueeze(-1)

        return super().forward(x)


class BjorckLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, iters=10, beta=0.5):

        super(BjorckLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.iters = iters
        self.beta = 0.5

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def _orthonormalize(self, iters=10, beta=0.5):
        """currently only supports order=1"""
        w = self.weight.t() / self._get_safe_bjorck_scaling()
        for _ in range(iters):
            w_t_w = w.t().mm(w)
            w = (1 + beta) * w - beta * w.mm(w_t_w)
        return w

    def _get_safe_bjorck_scaling(self):
        bjorck_scaling = torch.tensor(
            [np.sqrt(self.weight.size(0) * self.weight.size(1))],
            device=self.weight.device).float()
        return bjorck_scaling

    def forward(self, x):
        ortho_w = self._orthonormalize(
            self.iters,
            self.beta).t()

        return F.linear(x, ortho_w, self.bias)
