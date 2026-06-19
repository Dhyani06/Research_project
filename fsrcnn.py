import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSRCNN(nn.Module):
    """A compact implementation of the FSRCNN super-resolution model."""

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 3,
        d: int = 56,
        s: int = 12,
        m: int = 4,
    ):
        super().__init__()

        self.scale_factor = scale_factor
        self.num_channels = num_channels

        self.first = nn.Conv2d(num_channels, d, kernel_size=5, padding=2)
        self.first_act = nn.PReLU(d)

        self.shrink = nn.Conv2d(d, s, kernel_size=1)
        self.shrink_act = nn.PReLU(s)

        self.map_layers = nn.ModuleList(
            [
                nn.Conv2d(s, s, kernel_size=3, padding=1)
                for _ in range(m)
            ]
        )
        self.map_act = nn.ModuleList([nn.PReLU(s) for _ in range(m)])

        self.expand = nn.Conv2d(s, d, kernel_size=1)
        self.expand_act = nn.PReLU(d)

        # Deconvolution layer to upsample the features back to HR space
        self.deconv = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=4,
            output_padding=scale_factor - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_act(self.first(x))
        x = self.shrink_act(self.shrink(x))

        for conv, act in zip(self.map_layers, self.map_act):
            x = act(conv(x))

        x = self.expand_act(self.expand(x))
        x = self.deconv(x)
        return x


def init_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            # Paper: initialized with a normal distribution with mean 0 and standard deviation 0.001
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
