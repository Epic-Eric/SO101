import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    # Choose a valid number of groups that divides num_channels.
    # Bias toward 32 groups when possible, otherwise fall back.
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class EncoderStrong(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 128, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.stem = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        self.b1 = ResBlock(c)
        self.d1 = Downsample(c, c * 2)  # 64 -> 32

        self.b2 = ResBlock(c * 2)
        self.d2 = Downsample(c * 2, c * 4)  # 32 -> 16

        self.b3 = ResBlock(c * 4)
        self.d3 = Downsample(c * 4, c * 8)  # 16 -> 8

        self.b4 = ResBlock(c * 8)
        self.d4 = Downsample(c * 8, c * 8)  # 8 -> 4
        self.b5 = ResBlock(c * 8)

        self.norm = _group_norm(c * 8)
        self.fc = nn.Linear((c * 8) * 4 * 4, c * 8)
        self.fc_mu = nn.Linear(c * 8, latent_dim)
        self.fc_logvar = nn.Linear(c * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.b1(h)
        h = self.d1(h)

        h = self.b2(h)
        h = self.d2(h)

        h = self.b3(h)
        h = self.d3(h)

        h = self.b4(h)
        h = self.d4(h)
        h = self.b5(h)

        h = F.silu(self.norm(h))
        h = torch.flatten(h, 1)
        feat = F.silu(self.fc(h))
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar, feat


class DecoderStrong(nn.Module):
    def __init__(self, out_channels: int = 3, latent_dim: int = 128, base_channels: int = 64, output_activation: str = "tanh"):
        super().__init__()
        c = base_channels
        self._c = c
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, c * 8),
            nn.SiLU(),
            nn.Linear(c * 8, (c * 8) * 4 * 4),
        )
        self.u1 = Upsample(c * 8, c * 8)  # 4 -> 8
        self.b1 = ResBlock(c * 8)

        self.u2 = Upsample(c * 8, c * 4)  # 8 -> 16
        self.b2 = ResBlock(c * 4)

        self.u3 = Upsample(c * 4, c * 2)  # 16 -> 32
        self.b3 = ResBlock(c * 2)

        self.u4 = Upsample(c * 2, c)  # 32 -> 64
        self.b4 = ResBlock(c)

        self.norm = _group_norm(c)
        self.out = nn.Conv2d(c, out_channels, kernel_size=3, padding=1)

        if output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif output_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self._c * 8, 4, 4)
        h = self.u1(h)
        h = self.b1(h)

        h = self.u2(h)
        h = self.b2(h)

        h = self.u3(h)
        h = self.b3(h)

        h = self.u4(h)
        h = self.b4(h)

        h = F.silu(self.norm(h))
        x = self.out(h)
        return self.out_act(x)


class VAEStrong(nn.Module):
    """A stronger 64x64 VAE intended to be used as the image-latent model for a world model.

    - Encoder returns (mu, logvar, feat)
    - Decoder reconstructs from z
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        base_channels: int = 64,
        output_activation: str = "tanh",
        rec_loss: str = "mse",
        min_logvar: float = -12.0,
        max_logvar: float = 6.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = EncoderStrong(in_channels=in_channels, latent_dim=latent_dim, base_channels=base_channels)
        self.decoder = DecoderStrong(out_channels=in_channels, latent_dim=latent_dim, base_channels=base_channels, output_activation=output_activation)
        self.rec_loss_type = rec_loss
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, feat = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar, feat

    def reconstruction_loss(self, x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
        if self.rec_loss_type == "bce":
            return F.binary_cross_entropy(x_rec, x, reduction="sum") / x.size(0)
        if self.rec_loss_type == "mse":
            return F.mse_loss(x_rec, x, reduction="sum") / x.size(0)
        raise ValueError(f"Unknown rec_loss type: {self.rec_loss_type}")
