from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.src.models.rssm import RSSM
from model.src.models.vae_strong import VAEStrong


@dataclass
class WorldModelOutput:
    loss: torch.Tensor
    rec_loss: torch.Tensor
    kld: torch.Tensor
    kld_raw: torch.Tensor
    one_step_mse: torch.Tensor
    rollout_mse: torch.Tensor
    latent_drift: torch.Tensor
    kl_beta: torch.Tensor
    x_rec: torch.Tensor  # (B,T,C,H,W)


def _standard_normal_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q||N(0,1)) for diagonal Gaussian. Returns (B,)"""
    # 0.5 * (mu^2 + var - 1 - logvar)
    return 0.5 * (mu.pow(2) + torch.exp(logvar) - 1.0 - logvar).sum(dim=-1)


class WorldModel(nn.Module):
    """VAE + RSSM world model.

    - VAE provides per-frame stochastic latents z_t for images.
    - RSSM predicts a prior over z_t given past z and actions.

    Training objective is a sequence ELBO:
      recon(x_t | z_t) + beta * KL(q(z_t|x_t) || p(z_t|h_t))
    with t=0 KL against N(0,1).
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 128,
        deter_dim: int = 256,
        base_channels: int = 64,
        rec_loss: str = "mse",
        output_activation: str = "tanh",
        kl_beta: float = 1.0,
        free_nats: float = 0.0,
        min_std: float = 0.1,
        rssm_gate_threshold: float = 0.25,
        short_roll_horizon: int = 3,
    ):
        super().__init__()
        self.kl_beta = float(kl_beta)
        self.free_nats = float(free_nats)
        self.rssm_gate_threshold = float(rssm_gate_threshold)
        self.short_roll_horizon = int(short_roll_horizon)

        self.vae = VAEStrong(
            in_channels=3,
            latent_dim=latent_dim,
            base_channels=base_channels,
            output_activation=output_activation,
            rec_loss=rec_loss,
        )

        # Use VAE's encoder feature size as the RSSM observation embedding.
        # EncoderStrong outputs feat with dimension base_channels*8.
        obs_embed_dim = base_channels * 8
        self.rssm = RSSM(
            action_dim=action_dim,
            stoch_dim=latent_dim,
            deter_dim=deter_dim,
            obs_embed_dim=obs_embed_dim,
            hidden_dim=max(256, deter_dim),
            min_std=min_std,
        )

    def forward(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        kl_beta_override: float | None = None,
        rssm_gate_threshold: float | None = None,
        short_roll_horizon: int | None = None,
    ) -> WorldModelOutput:
        """Compute losses for a batch.

        images:  (B,T,C,H,W)
        actions: (B,T-1,A)
        """
        if images.dim() != 5:
            raise ValueError(f"Expected images (B,T,C,H,W), got {tuple(images.shape)}")
        if actions.dim() != 3:
            raise ValueError(f"Expected actions (B,T-1,A), got {tuple(actions.shape)}")
        b, t, c, h, w = images.shape
        if t < 2:
            raise ValueError("Need sequence length >= 2")
        if actions.shape[0] != b or actions.shape[1] != t - 1:
            raise ValueError("Actions must have shape (B, T-1, A)")

        flat = images.reshape(b * t, c, h, w)
        x_rec_flat, mu_flat, logvar_flat, feat_flat = self.vae(flat)
        x_rec = x_rec_flat.reshape(b, t, c, h, w)
        mu = mu_flat.reshape(b, t, -1)
        logvar = logvar_flat.reshape(b, t, -1)
        feat = feat_flat.reshape(b, t, -1)

        # Reconstruction loss is computed per-frame like the standalone VAE.
        rec_loss = self.vae.reconstruction_loss(flat, x_rec_flat)

        # RSSM KL loss
        state = self.rssm.init_state(b, device=images.device)
        state = type(state)(h=state.h, z=mu[:, 0])
        kls = []
        kls_raw = []
        one_step_errors = []
        latent_diffs = []
        states_for_rollout = [state]
        gate_tau = float(self.rssm_gate_threshold if rssm_gate_threshold is None else rssm_gate_threshold)
        roll_h = int(self.short_roll_horizon if short_roll_horizon is None else short_roll_horizon)

        # t = 0 KL against standard normal
        kls.append(_standard_normal_kl(mu[:, 0], logvar[:, 0]))
        kls_raw.append(kls[-1])

        for i in range(1, t):
            prev_mean = mu[:, i - 1]

            with torch.no_grad():
                gate_x = self.rssm.inp(torch.cat([prev_mean.detach(), actions[:, i - 1]], dim=-1))
                gate_h = self.rssm.gru(gate_x, state.h.detach())
                gate_prior_mu, gate_prior_logvar = self.rssm.prior_params(gate_h)
                one_step_err = F.mse_loss(gate_prior_mu, mu[:, i], reduction="none").mean(dim=-1)
            gate_mask = one_step_err > gate_tau
            z_dyn = torch.where(gate_mask.unsqueeze(-1), prev_mean.detach(), prev_mean)

            x = self.rssm.inp(torch.cat([z_dyn, actions[:, i - 1]], dim=-1))
            h_next = self.rssm.gru(x, state.h)
            prior_mu, prior_logvar = self.rssm.prior_params(h_next)

            post_mu, post_logvar = mu[:, i], logvar[:, i]
            kl_i = self.rssm.kl(post_mu, post_logvar, prior_mu, prior_logvar)
            kls.append(kl_i)
            kls_raw.append(kl_i.detach())

            # advance state deterministically; stochastic sample used for next step input only
            state = type(state)(h=h_next, z=post_mu)
            states_for_rollout.append(type(state)(h=h_next.detach(), z=post_mu.detach()))
            one_step_errors.append(one_step_err)
            latent_diffs.append((mu[:, i] - mu[:, i - 1]).abs().mean(dim=-1))

        kls_tensor = torch.stack(kls, dim=1)
        kld_raw = kls_tensor.mean()
        kld = kld_raw  # average over time & batch
        if self.free_nats > 0:
            # free nats is applied per-step per-sample, then averaged
            kls_t = kls_tensor  # (B,T)
            kls_t = torch.clamp(kls_t, min=self.free_nats)
            kld = kls_t.mean()

        beta = torch.tensor(self.kl_beta if kl_beta_override is None else kl_beta_override, device=images.device)
        loss = rec_loss + beta * kld

        # 1-step latent prediction error (mean over batch/time)
        if one_step_errors:
            one_step_mse = torch.stack(one_step_errors, dim=1).mean()
        else:
            one_step_mse = torch.tensor(0.0, device=images.device)

        # latent drift metric E[|z_t - z_{t-1}|]
        if latent_diffs:
            latent_drift = torch.stack(latent_diffs, dim=1).mean()
        else:
            latent_drift = torch.tensor(0.0, device=images.device)

        # short-horizon rollout error in latent space
        rollout_errors = []
        if roll_h > 0:
            horizon = min(roll_h, max(1, t - 1))
            with torch.no_grad():
                for start in range(0, t - horizon):
                    roll_state = states_for_rollout[start]
                    roll_state = type(roll_state)(h=roll_state.h.detach(), z=roll_state.z.detach())
                    for k in range(1, horizon + 1):
                        roll_state, (mu_p, logvar_p) = self.rssm.step(roll_state, actions[:, start + k - 1])
                        target_mu = mu[:, start + k]
                        rollout_errors.append(F.mse_loss(mu_p, target_mu, reduction="none").mean(dim=-1))
                        roll_state = type(roll_state)(h=roll_state.h, z=mu_p)
        rollout_mse = torch.stack(rollout_errors, dim=0).mean() if rollout_errors else torch.tensor(0.0, device=images.device)

        return WorldModelOutput(
            loss=loss,
            rec_loss=rec_loss.detach(),
            kld=kld.detach(),
            kld_raw=kld_raw.detach(),
            one_step_mse=one_step_mse.detach(),
            rollout_mse=rollout_mse.detach(),
            latent_drift=latent_drift.detach(),
            kl_beta=beta.detach(),
            x_rec=x_rec.detach(),
        )

    @torch.no_grad()
    def imagine(self, start_image: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out imagined images from a single start image.

        start_image: (1,C,H,W)
        actions: (T-1,A)
        returns: (T,C,H,W) decoded images
        """
        if start_image.dim() != 4 or start_image.shape[0] != 1:
            raise ValueError("start_image must be (1,C,H,W)")
        if actions.dim() != 2:
            raise ValueError("actions must be (T-1,A)")

        x_rec0, mu0, logvar0, feat0 = self.vae(start_image)
        z0 = self.vae.reparameterize(mu0, logvar0)

        t = actions.shape[0] + 1
        imgs = [self.vae.decode(z0).squeeze(0)]

        state = self.rssm.init_state(batch_size=1, device=start_image.device)
        state = type(state)(h=state.h, z=z0)

        for i in range(1, t):
            state, (mu_p, logvar_p) = self.rssm.step(state, actions[i - 1].unsqueeze(0))
            imgs.append(self.vae.decode(state.z).squeeze(0))

        return torch.stack(imgs, dim=0)
