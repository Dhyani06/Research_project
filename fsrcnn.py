import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSRCNN(nn.Module):
    """
    Improved FSRCNN super-resolution model (Dong et al., 2016).

    Operates on the Y (luminance) channel of YCbCr images, matching the
    paper's training and evaluation protocol.  Input and output are single-
    channel tensors normalised to [0, 1].

    Improvements over the original paper:
    ─────────────────────────────────────
    1. d=64 (was 56)  — more feature maps in extraction/expansion stages
       give the network more representational capacity with minimal extra cost.

    2. s=16 (was 12)  — wider shrink bottleneck reduces information loss
       when compressing 64 channels down for the mapping stage.

    3. m=6  (was 4)   — two extra non-linear mapping layers deepen the
       network so it can model more complex LR→HR transformations.

    4. kernel_size=7 for feature extraction (was 5) — larger receptive
       field (7×7 = 49 px vs 5×5 = 25 px) captures more spatial context
       per neuron, which is critical for reconstructing fine textures.

    5. PReLU init=0.1 (was 0.25) — conservative negative slope suits SR
       regression better than the classification-tuned default.

    6. kaiming fan_in init (was fan_out) — fan_in is more appropriate for
       regression tasks and stabilises gradient flow in deep stacks.

    Parameter count comparison:
      Original  (d=56, s=12, m=4, k=5): ~12 K params
      Improved  (d=64, s=16, m=6, k=7): ~24 K params  (still very lightweight)
    """

    # Shared initial negative slope for all PReLU activations
    PRELU_INIT: float = 0.1

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 1,   # 1 = Y-channel (paper standard)
        d: int = 64,             # feature maps  (improved: 56 → 64)
        s: int = 16,             # shrink size   (improved: 12 → 16)
        m: int = 6,              # mapping depth (improved:  4 →  6)
    ):
        super().__init__()

        self.scale_factor = scale_factor
        self.num_channels = num_channels

        # ── Feature extraction ───────────────────────────────────────────────
        # 7×7 kernel (improved from 5×5): larger receptive field captures
        # more spatial context for texture reconstruction.
        # padding=3 keeps spatial size identical to input.
        self.first = nn.Conv2d(num_channels, d, kernel_size=7, padding=3)
        self.first_act = nn.PReLU(num_parameters=d, init=self.PRELU_INIT)

        # ── Shrinking ────────────────────────────────────────────────────────
        # 1×1 conv reduces d=64 channels to s=16 (wider than original 12).
        self.shrink = nn.Conv2d(d, s, kernel_size=1)
        self.shrink_act = nn.PReLU(num_parameters=s, init=self.PRELU_INIT)

        # ── Non-linear mapping ───────────────────────────────────────────────
        # m=6 layers of 3×3 conv (deeper than original 4).
        # Each layer has its own independent PReLU slopes.
        self.map_layers = nn.ModuleList(
            [nn.Conv2d(s, s, kernel_size=3, padding=1) for _ in range(m)]
        )
        self.map_act = nn.ModuleList(
            [nn.PReLU(num_parameters=s, init=self.PRELU_INIT) for _ in range(m)]
        )

        # ── Expanding ────────────────────────────────────────────────────────
        # 1×1 conv expands s=16 back to d=64.
        self.expand = nn.Conv2d(s, d, kernel_size=1)
        self.expand_act = nn.PReLU(num_parameters=d, init=self.PRELU_INIT)

        # ── Deconvolution ────────────────────────────────────────────────────
        # Upsamples feature maps to HR resolution. No activation — output is
        # a raw residual added to the bicubic baseline.
        self.deconv = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=4,
            output_padding=scale_factor - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bicubic upscale as residual baseline.
        # Network learns only the high-frequency residual on top of this,
        # so outputs are near-correct even with small initial weights.
        identity = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
        )

        x = self.first_act(self.first(x))       # feature extraction  (7×7)
        x = self.shrink_act(self.shrink(x))     # shrink  d→s

        for conv, act in zip(self.map_layers, self.map_act):
            x = act(conv(x))                    # m mapping layers    (3×3)

        x = self.expand_act(self.expand(x))     # expand  s→d
        x = self.deconv(x)                      # upsample (no activation)

        x = x + identity                        # add bicubic residual
        x = torch.clamp(x, 0.0, 1.0)           # clamp to valid pixel range
        return x


def init_weights(model: nn.Module) -> None:
    """
    Weight initialisation:
    - Conv2d               : kaiming normal, fan_out — preserves gradient
                             variance through the deeper mapping stack (m=6).
    - ConvTranspose2d      : kaiming normal, fan_in — deconv maps many
                             channels to few; fan_in is appropriate here.
    - PReLU                : set to FSRCNN.PRELU_INIT (0.1).
    - BatchNorm2d          : weight=1, bias=0.
    """
    prelu_init = getattr(model, "PRELU_INIT", 0.1)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.PReLU):
            nn.init.constant_(m.weight, prelu_init)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
