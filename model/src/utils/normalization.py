from typing import Tuple

from model.src.interfaces.dataset import NormalizationParams


def get_default_normalization() -> NormalizationParams:
    """Returns default mean/std for simple [0,1] images centered to [-1,1]."""
    return NormalizationParams(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


def denormalize(tensor, norm: NormalizationParams):
    """Invert Normalize: x * std + mean, channel-wise."""
    import torch

    if tensor.dim() == 4:
        # B,C,H,W
        mean = torch.tensor(norm.mean, device=tensor.device).view(1, -1, 1, 1)
        std = torch.tensor(norm.std, device=tensor.device).view(1, -1, 1, 1)
    elif tensor.dim() == 3:
        # C,H,W
        mean = torch.tensor(norm.mean, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(norm.std, device=tensor.device).view(-1, 1, 1)
    else:
        raise ValueError("Expected tensor with 3 or 4 dims for denormalize")

    return tensor * std + mean
