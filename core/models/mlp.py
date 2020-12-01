import torch
import torch.nn as nn
import typing
import numpy as np


import pytorch_utils.nn as mynn


from .utils import BjorckLinear, RowNormalizedLinear, SphereProjection, Tanh


class Mlp(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_blocks: int, 
                 batch_norm: bool,
                 activation: str, 
                 compactification: str,
                 latent_dim: typing.Optional[int] = None):

        super().__init__()

        assert compactification in ['none', 'sphere_l2', 'tanh']

        assert activation in ['leaky_relu', 'relu']
        if activation == 'leaky_relu':
            def activation(): return nn.LeakyReLU(0.1)
        elif activation == 'relu':
            def activation(): return nn.ReLU()

        if batch_norm:
            def bn_1d(dim): return nn.BatchNorm1d(dim)
        else:
            def bn_1d(dim): return nn.Identity(dim)

        assert compactification in ['none', 'sphere_l2', 'tanh']
        tmp = {
            'none': nn.Identity(),
            'sphere_l2': SphereProjection(2),
            'tahn': Tanh()
        }
        compactifier = tmp[compactification]

        tmp = [mynn.LinearView()]
        input_dim = 3*32*32
        q = 1./np.sqrt(2).item()

        for i in range(num_blocks):

            dim_in = int(input_dim * q**i)
            dim_out = int(input_dim * q**(i+1))

            tmp += [
                nn.Linear(dim_in, dim_out),
                bn_1d(dim_out),
                activation()
            ]

        tmp.append(
            nn.Identity() if latent_dim is None else
            nn.Linear(int(input_dim * q**num_blocks), latent_dim)
        )

        tmp.append(compactifier)

        self.feat_ext = nn.Sequential(
            *tmp
        )

        latent_dim = int(input_dim * q**(num_blocks)) if latent_dim is None \
            else latent_dim

        self.cls = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.feat_ext(x)

        y_hat = self.cls(z)

        return y_hat, z
