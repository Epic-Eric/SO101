"""Action Encoder module for learning action embeddings.

This module provides learnable action embeddings to improve action conditioning
in latent world models. Instead of using raw actions directly, we embed them
through an MLP with LayerNorm and GELU activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    """Learnable action encoder that maps raw actions to embeddings.
    
    Uses an MLP with LayerNorm and GELU activation to create more expressive
    action representations that can be better utilized by the dynamics model.
    
    Args:
        action_dim: Dimensionality of input action vector
        embed_dim: Dimensionality of output action embedding (default: same as action_dim)
        hidden_dim: Dimensionality of hidden layer (default: 2x action_dim)
        num_layers: Number of hidden layers (default: 2)
    """
    
    def __init__(
        self,
        action_dim: int,
        embed_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.embed_dim = int(embed_dim) if embed_dim is not None else self.action_dim
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.action_dim * 2
        self.num_layers = int(num_layers)
        
        # Build MLP with LayerNorm and GELU
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.action_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.GELU())
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.GELU())
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.embed_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """Encode actions to embeddings.
        
        Args:
            action: Raw action tensor of shape (..., action_dim)
            
        Returns:
            Action embeddings of shape (..., embed_dim)
        """
        return self.encoder(action)
