import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, latent_dim: int = 128, output_activation: str = "sigmoid"):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # 32x32 -> 64x64
        )
        if output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif output_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        x_rec = self.deconv(h)
        x_rec = self.out_act(x_rec)
        return x_rec


class VAESimple(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        output_activation: str = "sigmoid",
        rec_loss: str = "bce",
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim, output_activation=output_activation)
        self.rec_loss_type = rec_loss

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)
        return x_rec, mu, logvar

    def loss_fn(self, x, x_rec, mu, logvar):
        # Reconstruction loss and KL divergence
        if self.rec_loss_type == "bce":
            rec_loss = F.binary_cross_entropy(x_rec, x, reduction="sum") / x.size(0)
        elif self.rec_loss_type == "mse":
            rec_loss = F.mse_loss(x_rec, x, reduction="sum") / x.size(0)
        else:
            raise ValueError(f"Unknown rec_loss type: {self.rec_loss_type}")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return rec_loss + kld, rec_loss, kld


# Backwards compatible alias
VAE = VAESimple
