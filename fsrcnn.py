import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSRCNN(nn.Module):
    """
    FSRCNN super-resolution model (Dong et al., 2016).

    Operates on the Y (luminance) channel of YCbCr images, matching the
    paper's training and evaluation protocol.  Input and output are single-
    channel tensors normalised to [0, 1].
    """

    def __init__(
        self,
        scale_factor: int = 2,
        # num_channels is kept at 1 (Y-channel) to match the paper.
        # Pass num_channels=3 only if you intentionally want RGB training.
        num_channels: int = 1,
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
        # Bicubic upscale as residual baseline — the network learns only the
        # high-frequency residual.  This ensures outputs start near the correct
        # pixel range even with near-zero weights, fixing the vanishing-output
        # problem without requiring special initialisation tricks.
        identity = torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
        )

        x = self.first_act(self.first(x))
        x = self.shrink_act(self.shrink(x))

        for conv, act in zip(self.map_layers, self.map_act):
            x = act(conv(x))

        x = self.expand_act(self.expand(x))
        x = self.deconv(x)

        # Add residual and clamp to valid range
        x = x + identity
        x = torch.clamp(x, 0.0, 1.0)
        return x


def init_weights(model: nn.Module) -> None:
    """
    Initialise weights:
    - Conv2d / ConvTranspose2d : kaiming normal
    - PReLU                    : default (0.25)
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
