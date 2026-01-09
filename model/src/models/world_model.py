from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.src.models.rssm import RSSM, RSSMState
from model.src.models.vae_strong import VAEStrong
from model.src.models.action_encoder import ActionEncoder
from model.src.models.contrastive_loss import ContrastiveActionLoss


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
    # New metrics for action conditioning
    contrastive_loss: torch.Tensor
    action_sensitivity: torch.Tensor
    latent_action_variance: torch.Tensor


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
        # New parameters for action conditioning
        use_action_encoder: bool = True,
        action_embed_dim: int = None,
        contrastive_weight: float = 0.1,
        contrastive_margin: float = 1.0,
        grad_detach_schedule_k: int = 4,
    ):
        super().__init__()
        self.kl_beta = float(kl_beta)
        self.free_nats = float(free_nats)
        self.rssm_gate_threshold = float(rssm_gate_threshold)
        self.short_roll_horizon = int(short_roll_horizon)
        self.use_action_encoder = bool(use_action_encoder)
        self.contrastive_weight = float(contrastive_weight)
        self.grad_detach_schedule_k = int(grad_detach_schedule_k)
        self._step_counter = 0  # Track training steps for gradient detachment

        self.vae = VAEStrong(
            in_channels=3,
            latent_dim=latent_dim,
            base_channels=base_channels,
            output_activation=output_activation,
            rec_loss=rec_loss,
        )

        # Action encoder
        if self.use_action_encoder:
            self.action_encoder = ActionEncoder(
                action_dim=action_dim,
                embed_dim=action_embed_dim if action_embed_dim is not None else action_dim,
                hidden_dim=action_dim * 2,
                num_layers=2,
            )
            action_input_dim = self.action_encoder.embed_dim
        else:
            self.action_encoder = None
            action_input_dim = action_dim

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
            action_embed_dim=action_input_dim,
        )
        
        # Contrastive loss for action sensitivity
        self.contrastive_loss_fn = ContrastiveActionLoss(
            margin=contrastive_margin,
            distance_type="l2",
        )

    def _make_state(self, h: torch.Tensor, z: torch.Tensor) -> RSSMState:
        """Centralize RSSMState construction so call sites remain consistent if the state schema evolves."""
        return RSSMState(h=h, z=z)

    def forward(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        kl_beta_override: Optional[float] = None,
        rssm_gate_threshold: Optional[float] = None,
        short_roll_horizon: Optional[int] = None,
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
        
        # Scheduled gradient detachment between VAE encoder and RSSM
        # Increment step counter in training mode
        if self.training:
            self._step_counter += 1
        allow_vae_grad = (self._step_counter % self.grad_detach_schedule_k == 0) if self.training else True
        
        x_rec_flat, mu_flat, logvar_flat, feat_flat = self.vae(flat)
        x_rec = x_rec_flat.reshape(b, t, c, h, w)
        mu_raw = mu_flat.reshape(b, t, -1)
        logvar = logvar_flat.reshape(b, t, -1)
        feat = feat_flat.reshape(b, t, -1)
        
        # Apply gradient detachment if scheduled
        if allow_vae_grad:
            mu = mu_raw
        else:
            mu = mu_raw.detach()

        # Encode actions to embeddings if using action encoder
        if self.use_action_encoder:
            action_input = self.action_encoder(actions)
        else:
            action_input = actions

        # Reconstruction loss is computed per-frame like the standalone VAE.
        rec_loss = self.vae.reconstruction_loss(flat, x_rec_flat)

        # RSSM KL loss
        state = self.rssm.init_state(b, device=images.device)
        state = self._make_state(state.h, mu[:, 0])
        kls = []
        kls_raw = []
        one_step_errors = []
        latent_diffs = []
        state_for_rollout = self._make_state(state.h.detach(), mu[:, 0].detach())
        states_for_rollout = [state_for_rollout]
        gate_tau = float(self.rssm_gate_threshold if rssm_gate_threshold is None else rssm_gate_threshold)
        rollout_horizon = int(self.short_roll_horizon if short_roll_horizon is None else short_roll_horizon)

        # t = 0 KL against standard normal
        kls.append(_standard_normal_kl(mu[:, 0], logvar[:, 0]))
        kls_raw.append(kls[-1])

        for i in range(1, t):
            prev_mean = mu[:, i - 1]

            # Provisional prediction without gating to compute gate mask
            gate_x = self.rssm.inp(torch.cat([prev_mean, action_input[:, i - 1]], dim=-1))
            gate_h = self.rssm.gru(gate_x, state.h)
            gate_prior_mu, gate_prior_logvar = self.rssm.prior_params(gate_h)
            gate_one_step_err = F.mse_loss(gate_prior_mu, mu[:, i], reduction="none").mean(dim=-1)
            gate_mask = gate_one_step_err > gate_tau

            # Apply gating by stopping encoder gradients when gate_mask is True
            z_for_dynamics = prev_mean + (prev_mean.detach() - prev_mean) * gate_mask.unsqueeze(-1)
            x = self.rssm.inp(torch.cat([z_for_dynamics, action_input[:, i - 1]], dim=-1))
            h_next = self.rssm.gru(x, state.h)
            prior_mu, prior_logvar = self.rssm.prior_params(h_next)
            one_step_err = F.mse_loss(prior_mu, mu[:, i], reduction="none").mean(dim=-1)
            gate_mask_final = one_step_err > gate_tau
            if not torch.equal(gate_mask_final, gate_mask):
                gate_mask = gate_mask_final
                z_for_dynamics = prev_mean + (prev_mean.detach() - prev_mean) * gate_mask.unsqueeze(-1)
                x = self.rssm.inp(torch.cat([z_for_dynamics, action_input[:, i - 1]], dim=-1))
                h_next = self.rssm.gru(x, state.h)
                prior_mu, prior_logvar = self.rssm.prior_params(h_next)
                one_step_err = F.mse_loss(prior_mu, mu[:, i], reduction="none").mean(dim=-1)

            post_mu, post_logvar = mu[:, i], logvar[:, i]
            kl_i = self.rssm.kl(post_mu, post_logvar, prior_mu, prior_logvar)
            kls.append(kl_i)
            kls_raw.append(self.rssm.kl(post_mu, post_logvar, prior_mu.detach(), prior_logvar.detach()))

            # advance state deterministically; stochastic sample used for next step input only
            state = self._make_state(h_next, post_mu)
            states_for_rollout.append(self._make_state(h_next.detach(), post_mu.detach()))
            one_step_errors.append(one_step_err)
            latent_diffs.append((mu[:, i] - mu[:, i - 1]).abs().mean(dim=-1))

        kls_tensor = torch.stack(kls, dim=1)
        kld_raw = kls_tensor.mean()
        kls_clamped = torch.clamp(kls_tensor, min=self.free_nats) if self.free_nats > 0 else kls_tensor
        kld = kls_clamped.mean()

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
        if rollout_horizon > 0:
            # t >= 2 is enforced above; horizon is capped accordingly
            horizon = min(rollout_horizon, max(1, t - 1))
            with torch.no_grad():
                # horizon is capped at (t-1), so t - horizon is at least 1 when t >= 2
                for start in range(0, t - horizon):
                    roll_state = states_for_rollout[start]
                    roll_state = self._make_state(roll_state.h.detach(), roll_state.z.detach())
                    for k in range(1, horizon + 1):
                        roll_state, (mu_p, logvar_p) = self.rssm.step(roll_state, action_input[:, start + k - 1])
                        target_mu = mu[:, start + k]
                        rollout_errors.append(F.mse_loss(mu_p, target_mu, reduction="none").mean(dim=-1))
                        roll_state = self._make_state(roll_state.h, mu_p)
        rollout_mse = torch.stack(rollout_errors, dim=0).mean() if rollout_errors else torch.tensor(0.0, device=images.device)

        # Compute contrastive action sensitivity loss
        contrastive_loss = torch.tensor(0.0, device=images.device)
        action_sensitivity = torch.tensor(0.0, device=images.device)
        latent_action_variance = torch.tensor(0.0, device=images.device)
        
        if self.training and t > 1 and b > 1:
            # Sample random time steps and batch indices for contrastive pairs
            # For each state, predict next state with two different actions
            num_samples = min(b, 4)  # Limit samples for efficiency
            sample_indices = torch.randperm(b, device=images.device)[:num_samples]
            
            contrastive_losses = []
            action_sensitivities = []
            action_variances = []
            
            for idx in sample_indices:
                # Pick a random time step
                if t > 2:
                    time_idx = torch.randint(0, t - 1, (1,), device=images.device).item()
                else:
                    time_idx = 0
                
                # Current state
                curr_z = mu[idx:idx+1, time_idx]
                curr_h = states_for_rollout[time_idx].h[idx:idx+1]
                curr_state = self._make_state(curr_h, curr_z)
                
                # Two different actions from the batch
                action_a = action_input[idx:idx+1, time_idx]
                # Pick a different action from another batch element
                other_idx = (idx + 1) % b
                action_b = action_input[other_idx:other_idx+1, time_idx]
                
                # Predict next states with different actions
                state_a, (mu_a, _) = self.rssm.step(curr_state, action_a)
                state_b, (mu_b, _) = self.rssm.step(curr_state, action_b)
                
                # Contrastive loss: encourage predictions to differ
                contrast_loss = self.contrastive_loss_fn.pairwise_loss(mu_a, mu_b)
                contrastive_losses.append(contrast_loss)
                
                # Compute action sensitivity (variance in latent space)
                latent_var = torch.var(torch.cat([mu_a, mu_b], dim=0), dim=0).mean()
                action_variances.append(latent_var)
            
            if contrastive_losses:
                contrastive_loss = torch.stack(contrastive_losses).mean()
                latent_action_variance = torch.stack(action_variances).mean()
        
        # Add contrastive loss to total loss
        loss = loss + self.contrastive_weight * contrastive_loss

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
            contrastive_loss=contrastive_loss.detach(),
            action_sensitivity=action_sensitivity.detach(),
            latent_action_variance=latent_action_variance.detach(),
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

        # Encode actions if using action encoder
        if self.use_action_encoder:
            action_input = self.action_encoder(actions)
        else:
            action_input = actions

        state = self.rssm.init_state(batch_size=1, device=start_image.device)
        state = self._make_state(state.h, z0)

        for i in range(1, t):
            state, (mu_p, logvar_p) = self.rssm.step(state, action_input[i - 1].unsqueeze(0))
            imgs.append(self.vae.decode(state.z).squeeze(0))

        return torch.stack(imgs, dim=0)
