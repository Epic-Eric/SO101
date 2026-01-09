from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    h: torch.Tensor  # deterministic state, (B, deter_dim)
    z: torch.Tensor  # stochastic sample, (B, stoch_dim)


def _gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """KL(q||p) for diagonal Gaussians. Returns (B,) with sum over dim."""
    # 0.5 * ( log(sigma_p^2 / sigma_q^2) + (sigma_q^2 + (mu_q-mu_p)^2)/sigma_p^2 - 1 )
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8) - 1.0)
    return kl.sum(dim=-1)


class RSSM(nn.Module):
    """A small Dreamer-style Recurrent State Space Model.

    This RSSM models a stochastic latent z_t with a deterministic recurrent state h_t.

    - Prior:   p(z_t | h_t)
    - Posterior: q(z_t | h_t, o_embed_t)

    The recurrent update uses (z_{t-1}, a_{t-1}) as input.
    """

    def __init__(
        self,
        action_dim: int,
        stoch_dim: int = 128,
        deter_dim: int = 256,
        obs_embed_dim: int = 512,
        hidden_dim: int = 256,
        min_std: float = 0.1,
        action_embed_dim: int = None,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.stoch_dim = int(stoch_dim)
        self.deter_dim = int(deter_dim)
        self.obs_embed_dim = int(obs_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_std = float(min_std)
        # If action_embed_dim is provided, expect action embeddings; otherwise use raw actions
        self.action_embed_dim = int(action_embed_dim) if action_embed_dim is not None else self.action_dim

        self.inp = nn.Sequential(
            nn.Linear(self.stoch_dim + self.action_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.deter_dim)

        self.prior = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.stoch_dim),
        )

        self.post = nn.Sequential(
            nn.Linear(self.deter_dim + self.obs_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.stoch_dim),
        )

    def init_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        dev = torch.device(device)
        h = torch.zeros(batch_size, self.deter_dim, device=dev)
        z = torch.zeros(batch_size, self.stoch_dim, device=dev)
        return RSSMState(h=h, z=z)

    def _split_params(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, raw_std = torch.chunk(params, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        logvar = 2.0 * torch.log(std)
        return mu, logvar

    def prior_params(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_params(self.prior(h))

    def posterior_params(self, h: torch.Tensor, obs_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_params(self.post(torch.cat([h, obs_embed], dim=-1)))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def step(self, prev: RSSMState, action: torch.Tensor) -> tuple[RSSMState, tuple[torch.Tensor, torch.Tensor]]:
        """One transition step using prior only.

        Returns next state and (prior_mu, prior_logvar) for z_t.
        """
        x = self.inp(torch.cat([prev.z, action], dim=-1))
        h = self.gru(x, prev.h)
        mu, logvar = self.prior_params(h)
        z = self.reparameterize(mu, logvar)
        return RSSMState(h=h, z=z), (mu, logvar)

    def kl(self, mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
        return _gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
