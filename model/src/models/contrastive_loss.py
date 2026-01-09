"""Contrastive action sensitivity loss for world models.

This module implements a contrastive loss that encourages different actions
to produce measurably different next-state predictions, addressing the
action collapse problem in latent world models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveActionLoss(nn.Module):
    """Contrastive loss to enforce action sensitivity in latent transitions.
    
    For the same state (z_t, h_t) with different actions, this loss encourages
    the predicted next states z_{t+1} to be far apart, preventing action collapse.
    
    Args:
        margin: Minimum desired distance between predictions for different actions
        temperature: Temperature for similarity scaling (optional)
        distance_type: Type of distance metric ('l2', 'cosine')
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 1.0,
        distance_type: str = "l2",
    ):
        super().__init__()
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.distance_type = str(distance_type)
        
        if self.distance_type not in ["l2", "cosine"]:
            raise ValueError(f"distance_type must be 'l2' or 'cosine', got {self.distance_type}")
    
    def compute_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute distance between two latent states.
        
        Args:
            z1: First latent state (B, latent_dim)
            z2: Second latent state (B, latent_dim)
            
        Returns:
            Distance tensor (B,)
        """
        if self.distance_type == "l2":
            # Euclidean distance
            return torch.norm(z1 - z2, p=2, dim=-1)
        elif self.distance_type == "cosine":
            # Cosine distance (1 - cosine similarity)
            sim = F.cosine_similarity(z1, z2, dim=-1)
            return 1.0 - sim
    
    def forward(
        self,
        z_next_anchor: torch.Tensor,
        z_next_positive: torch.Tensor,
        z_next_negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Anchor and positive should be from similar/same actions,
        while negative should be from different actions.
        
        Args:
            z_next_anchor: Predicted next state for anchor action (B, latent_dim)
            z_next_positive: Predicted next state for positive action (B, latent_dim)  
            z_next_negative: Predicted next state for negative action (B, latent_dim)
            
        Returns:
            Contrastive loss scalar
        """
        # Distance between anchor and positive (should be small)
        d_pos = self.compute_distance(z_next_anchor, z_next_positive)
        
        # Distance between anchor and negative (should be large)
        d_neg = self.compute_distance(z_next_anchor, z_next_negative)
        
        # Triplet-style margin loss
        # loss = max(0, d_pos - d_neg + margin)
        loss = F.relu(d_pos - d_neg + self.margin)
        
        return loss.mean()
    
    def pairwise_loss(
        self,
        z_next_same: torch.Tensor,
        z_next_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified pairwise contrastive loss.
        
        For same state but different actions, encourage distance to be large.
        
        Args:
            z_next_same: Predicted next state for action A (B, latent_dim)
            z_next_diff: Predicted next state for action B (B, latent_dim)
            
        Returns:
            Contrastive loss scalar
        """
        # Distance between predictions with different actions (should be large)
        distance = self.compute_distance(z_next_same, z_next_diff)
        
        # Encourage distance to exceed margin
        # loss = max(0, margin - distance)
        loss = F.relu(self.margin - distance)
        
        return loss.mean()


def compute_action_sensitivity_norm(
    model,
    state,
    action: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Compute action sensitivity as gradient norm ||∂z_{t+1} / ∂a_t||.
    
    Args:
        model: RSSM model with step() method
        state: Current RSSMState
        action: Action tensor (B, action_dim), requires_grad=True
        epsilon: Small value for numerical stability
        
    Returns:
        Sensitivity norm (B,)
    """
    if not action.requires_grad:
        action = action.clone().detach().requires_grad_(True)
    
    # Forward pass
    next_state, (mu, logvar) = model.step(state, action)
    
    # Compute gradient of predicted latent w.r.t. action
    # Use mu (mean) as the output to differentiate
    grads = []
    for i in range(mu.shape[0]):
        if mu.shape[0] == 1:
            # Single batch element
            grad = torch.autograd.grad(
                outputs=mu[i].sum(),
                inputs=action,
                create_graph=False,
                retain_graph=(i < mu.shape[0] - 1),
            )[0][i]
        else:
            grad = torch.autograd.grad(
                outputs=mu[i].sum(),
                inputs=action,
                create_graph=False,
                retain_graph=(i < mu.shape[0] - 1),
            )[0][i]
        grads.append(grad)
    
    grads = torch.stack(grads, dim=0)
    sensitivity = torch.norm(grads, p=2, dim=-1)
    
    return sensitivity
