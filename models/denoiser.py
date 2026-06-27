"""CNN denoiser modules for use inside the unrolled reconstructor.

Two variants:
  - UNet:  encoder-decoder with skip connections; default for complex images.
  - DnCNN: residual flat CNN; lightweight alternative.

Both accept x ∈ R^{B × C_in × H × W} and return the same shape.
When precision_conditioning=True, the precision map ρ (broadcast to spatial dims)
is concatenated as an extra channel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# U-Net denoiser
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """3×3 Conv → BN → ReLU × 2."""
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDenoiser(nn.Module):
    """
    Encoder-decoder U-Net for complex MRI image denoising.

    in_channels:  C_in (2 real/imag + optional precision channel(s))
    out_channels: C_out (2 real/imag)
    features:     channel counts at each encoder level
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        features: List[int] = None,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f, use_batchnorm))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, use_batchnorm)
        ch = features[-1] * 2

        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f, use_batchnorm))
            ch = f

        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x
        for enc, pool in zip(self.encoders, self.pools):
            out = enc(out)
            skips.append(out)
            out = pool(out)

        out = self.bottleneck(out)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = upconv(out)
            # Pad if spatial dims differ (odd sizes)
            if out.shape != skip.shape:
                out = F.pad(out, [0, skip.shape[-1] - out.shape[-1],
                                   0, skip.shape[-2] - out.shape[-2]])
            out = torch.cat([skip, out], dim=1)
            out = dec(out)

        return self.head(out)


# ---------------------------------------------------------------------------
# DnCNN denoiser
# ---------------------------------------------------------------------------

class DnCNNDenoiser(nn.Module):
    """
    DnCNN-style residual denoiser (Zhang et al. 2017).
    15 layers, 3×3 kernels, BN+ReLU, residual connection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        n_layers: int = 15,
        n_features: int = 64,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = [nn.Conv2d(in_channels, n_features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 2):
            layers += [
                nn.Conv2d(n_features, n_features, 3, padding=1, bias=not use_batchnorm),
                nn.BatchNorm2d(n_features) if use_batchnorm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(n_features, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual: predict noise and subtract
        noise = self.net(x)
        return x[:, :self.out_channels] - noise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_denoiser(cfg: dict) -> nn.Module:
    """Build denoiser from recon config."""
    d = cfg.get("denoiser", {})
    dtype = d.get("type", "unet")
    in_ch = d.get("in_channels", 3)
    out_ch = d.get("out_channels", 2)
    use_bn = d.get("use_batchnorm", True)

    if dtype == "unet":
        features = d.get("features", [32, 64, 128, 256])
        return UNetDenoiser(in_ch, out_ch, features, use_bn)
    elif dtype == "dncnn":
        return DnCNNDenoiser(in_ch, out_ch, use_batchnorm=use_bn)
    else:
        raise ValueError(f"Unknown denoiser type: {dtype}")
