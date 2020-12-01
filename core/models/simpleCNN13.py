import torch
import torch.nn as nn
import typing

import pytorch_utils.nn as mynn

from .utils import BjorckLinear, RowNormalizedLinear, SphereProjection, Tanh


class SimpleCNN13(nn.Module):
    def __init__(self,
                 num_classes: int,
                 batch_norm: bool,
                 drop_out: bool,
                 final_bn: bool,
                 cls_norm: str,
                #  sigmoid: bool,
                 compactification: str, 
                 linear_bias: bool,
                 latent_dim: typing.Optional[int] = None
                 ):
        super().__init__()
        assert cls_norm in ['spectral', 'bjorck', 'row_norm', 'none']
        assert compactification in ['none', 'sphere_l2', 'tanh']

        def activation(): return nn.LeakyReLU(0.1)

        if drop_out:
            def dropout(p): return nn.Dropout(p)
        else:
            def dropout(p): return nn.Identity()

        bn_affine = True

        if batch_norm:
            def bn_2d(dim): return nn.BatchNorm2d(dim, affine=bn_affine)

            def bn_1d(dim): return nn.BatchNorm1d(dim, affine=bn_affine)
        else:
            def bn_2d(dim): return nn.Identity(dim)

            def bn_1d(dim): return nn.Identity(dim)

        assert compactification in ['none', 'sphere_l2', 'tanh']        
        tmp = {
            'none': nn.Identity(),
            'sphere_l2': SphereProjection(2),
            'tahn': Tanh()
        }
        compactifier = tmp[compactification]


        self.feat_ext = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            bn_2d(128),
            activation(),
            nn.Conv2d(128, 128, 3, padding=1),
            bn_2d(128),
            activation(),
            nn.Conv2d(128, 128, 3, padding=1),
            bn_2d(128),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            dropout(0.5),
            #
            nn.Conv2d(128, 256, 3, padding=1),
            bn_2d(256),
            activation(),
            nn.Conv2d(256, 256, 3, padding=1),
            bn_2d(256),
            activation(),
            nn.Conv2d(256, 256, 3, padding=1),
            bn_2d(256),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            dropout(0.5),
            #
            nn.Conv2d(256, 512, 3, padding=0),
            bn_2d(512),
            activation(),
            nn.Conv2d(512, 256, 1, padding=0),
            bn_2d(256),
            activation(),
            nn.Conv2d(256, 128, 1, padding=0),
            bn_2d(128) if final_bn else nn.Identity(),
            activation(),
            nn.AvgPool2d(6, stride=2, padding=0),
            mynn.LinearView(),
            nn.Identity() if latent_dim is None else nn.Linear(128, latent_dim),
            compactifier, 
        )            

        latent_dim = 128 if latent_dim is None else latent_dim

        if cls_norm == 'bjorck':
            cls = BjorckLinear(latent_dim, num_classes, bias=linear_bias)

        elif cls_norm == 'spectral':
            cls = nn.utils.spectral_norm(
                nn.Linear(latent_dim, num_classes, bias=linear_bias))

        elif cls_norm == 'row_norm':
            cls = RowNormalizedLinear(
                latent_dim, num_classes, bias=linear_bias)

        elif cls_norm == 'none':
            cls = nn.Linear(latent_dim, num_classes, bias=linear_bias)

        self.cls = cls

    def forward(self, x):
        z = self.feat_ext(x)

        y_hat = self.cls(z)

        return y_hat, z


class SimpleCNN13FrozenLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        cnn13 = SimpleCNN13(*args,  **kwargs)
        self.feat_ext = cnn13.feat_ext

        num_classes = cnn13.cls.weight.size(0)
        feat_dim = cnn13.cls.weight.size(1)
        w = torch.cat((torch.eye(num_classes), torch.zeros(num_classes, feat_dim - num_classes)), dim=1).float()
        w = w - w.mean(dim=0)
        w = w / w.norm(2, dim=-1, keepdim=True)

        num_classes = cnn13.cls.weight.size(0)
        feat_dim = cnn13.cls.weight.size(1)
        w = torch.cat(
            (
                torch.eye(num_classes), 
                torch.zeros(num_classes, feat_dim - num_classes)
            ), 
            dim=1).float()

        w = w - w.mean(dim=0)
        w = w / w.norm(2, dim=-1, keepdim=True)
        w = w.T
        w = w.unsqueeze(0)

        self.register_buffer('w_cls', w)

    def forward(self, x):

        z = self.feat_ext(x)
        
        y_hat = torch.matmul(z.unsqueeze(1), self.w_cls).squeeze()

        return y_hat, z
