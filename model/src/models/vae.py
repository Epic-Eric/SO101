"""VAE models.

This module keeps backwards compatibility with the original simple VAE used by
`train_model.py` / `model.src.core.train_vae`.

- `VAE` is the original implementation (now in `vae_simple.py`).
- `VAEStrong` is a more expressive VAE intended for world-model training.
"""

from model.src.models.vae_simple import VAE, VAESimple
from model.src.models.vae_strong import VAEStrong

__all__ = [
    "VAE",
    "VAESimple",
    "VAEStrong",
]
