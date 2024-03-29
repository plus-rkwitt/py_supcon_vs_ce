"""
ResNet from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py adapted for flexible calls number
"""

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''




import torch.nn.functional as F
import torch.nn as nn
import torch
import typing
import functools
import pytorch_utils.nn as mynn
from .utils import SphereProjection, Tanh, NormedLinear, FixedSphericalSimplexLinear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_norm=True):
        super(BasicBlock, self).__init__()

        batch_norm_factory = nn.BatchNorm2d if batch_norm else nn.Identity

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = batch_norm_factory(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = batch_norm_factory(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                batch_norm_factory(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, batch_norm=True):
        super(Bottleneck, self).__init__()

        batch_norm_factory = nn.BatchNorm2d if batch_norm else nn.Identity

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = batch_norm_factory(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = batch_norm_factory(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = batch_norm_factory(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                batch_norm_factory(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _check_cfg(cfg):
    assert isinstance(cfg, tuple) and len(cfg) == 2
    type_str, kwargs = cfg
    assert isinstance(type_str, str)
    assert isinstance(kwargs, dict)


class ResNet(nn.Module):

    compactifier_types = {
        'none': nn.Identity,
        'sphere_l2': functools.partial(SphereProjection, p=2, learn_radius=False, radius_init=1.),
        'tahn': Tanh,
        'sphere_l2_learned': functools.partial(SphereProjection, p=2, learn_radius=True)
    }

    linear_types = {
        'Linear': nn.Linear,
        'NormedLinear': NormedLinear, 
        'FixedSphericalSimplexLinear': FixedSphericalSimplexLinear
    }

    def __init__(self,
                 block: int,
                 num_blocks: list,
                 num_classes: int,
                 compactification_cfg: tuple,
                 linear_cfg: tuple,
                 batch_norm: bool,
                 latent_dim: typing.Optional[int]):
        super(ResNet, self).__init__()

        batch_norm_factory = nn.BatchNorm2d if batch_norm else nn.Identity

        self.in_planes = 64
        _check_cfg(compactification_cfg)
        _check_cfg(linear_cfg)

        comp_type, comp_kwargs = compactification_cfg
        compactifier = self.compactifier_types[comp_type](**comp_kwargs)

        self.feat_ext = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            batch_norm_factory(64),
            nn.ReLU(),
            self._make_layer(
                block, 64, num_blocks[0], stride=1, batch_norm=batch_norm),
            self._make_layer(
                block, 128, num_blocks[1], stride=2, batch_norm=batch_norm),
            self._make_layer(
                block, 256, num_blocks[2], stride=2, batch_norm=batch_norm),
            self._make_layer(
                block, 512, num_blocks[3], stride=2, batch_norm=batch_norm),
            nn.AvgPool2d(kernel_size=4),
            mynn.LinearView(),
            nn.Identity() if latent_dim is None else nn.Linear(
                512*block.expansion, latent_dim
            ),
            compactifier
        )

        lin_type, lin_kwargs = linear_cfg
        self.cls = self.linear_types[lin_type](
            512*block.expansion if latent_dim is None else latent_dim,
            num_classes,
            **lin_kwargs)

    def _make_layer(self, block, planes, num_blocks, stride, batch_norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, batch_norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.feat_ext(x)
        y_hat = self.cls(z)
        return y_hat, z


ResNet18 = functools.partial(ResNet, BasicBlock, [2, 2, 2, 2])

ResNet34 = functools.partial(ResNet, BasicBlock, [3, 4, 6, 3])

ResNet50 = functools.partial(ResNet, Bottleneck, [3, 4, 6, 3])

ResNet101 = functools.partial(ResNet, Bottleneck, [3, 4, 23, 3])

ResNet152 = functools.partial(ResNet, Bottleneck, [3, 8, 36, 3])
