"""Augmentation utilities for model training.

Provides CutMix and MixUp helpers that operate on batches of tensors.

Typical usage (CutMix):
    imgs, _, _, lam = cutmix(imgs, imgs, alpha=1.0)
    # For VAE training, use the mixed image as both input and reconstruction target.

The helpers assume inputs are torch tensors shaped (N, C, H, W).
"""
from __future__ import annotations

from typing import Tuple, Optional

import torch


def rand_bbox(size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
    """Generate a random bbox for CutMix.

    Args:
        size: tuple (N, C, H, W) of the batch tensor size — only H and W used.
        lam: cut ratio lambda coming from Beta(alpha, alpha)

    Returns:
        x1, y1, x2, y2 coordinates (inclusive/exclusive) of the rectangle to replace.
    """
    _, _, H, W = size
    # area ratio to cut
    cut_rat = float(torch.sqrt(torch.tensor(1.0 - lam)))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    return x1, y1, x2, y2


def cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    device: Optional[torch.device] = None,
):
    """Apply CutMix to a batch of images and targets.

    Args:
        images: float tensor (N, C, H, W)
        targets: tensor (N, ...) the same shape as images for self-supervised tasks or labels for supervised
        alpha: beta distribution parameter; alpha <= 0 disables CutMix
        device: optional device for sampling; defaults to images.device

    Returns:
        mixed_images: torch.Tensor (N, C, H, W)
        targets_a: original targets
        targets_b: shuffled targets
        lam: mixing coefficient (float)
    """
    if device is None:
        device = images.device

    N = images.size(0)
    if alpha <= 0:
        return images, targets, targets, 1.0

    # sample lambda from Beta(alpha, alpha)
    try:
        dist = torch.distributions.Beta(alpha, alpha)
        lam = float(dist.sample())
    except Exception:
        lam = 1.0

    rand_index = torch.randperm(N).to(device)

    mixed_images = images.clone()

    # compute bbox
    x1, y1, x2, y2 = rand_bbox(images.size(), lam)

    # replace region for all images in batch with shuffled images
    mixed_images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

    # Adjust lambda to exactly match pixel ratio (in case bbox clipped at borders)
    area = (x2 - x1) * (y2 - y1)
    lam_adjust = 1.0 - area / float(images.size(2) * images.size(3))

    targets_a = targets
    targets_b = targets[rand_index]
    return mixed_images, targets_a, targets_b, lam_adjust


def mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    device: Optional[torch.device] = None,
):
    """Apply MixUp to a batch of images and targets.

    Returns: mixed_images, targets_a, targets_b, lam
    """
    if device is None:
        device = images.device

    N = images.size(0)
    if alpha <= 0:
        return images, targets, targets, 1.0

    try:
        dist = torch.distributions.Beta(alpha, alpha)
        lam = float(dist.sample())
    except Exception:
        lam = 1.0

    rand_index = torch.randperm(N).to(device)
    mixed_images = lam * images + (1 - lam) * images[rand_index]

    targets_a = targets
    targets_b = targets[rand_index]
    return mixed_images, targets_a, targets_b, lam


def mix_criterion(criterion, preds: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float):
    """Helper to compute mixed loss given a base criterion and mixed targets.

    Computes: lam * criterion(preds, targets_a) + (1-lam) * criterion(preds, targets_b)
    """
    return lam * criterion(preds, targets_a) + (1.0 - lam) * criterion(preds, targets_b)
