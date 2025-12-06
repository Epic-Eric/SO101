import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_latent_hist(mu: torch.Tensor, logvar: torch.Tensor, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mu_np = mu.detach().cpu().numpy().ravel()
    lv_np = logvar.detach().cpu().numpy().ravel()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(mu_np, bins=50, alpha=0.8, color="steelblue")
    plt.title("Posterior mean (mu)")
    plt.subplot(1, 2, 2)
    plt.hist(lv_np, bins=50, alpha=0.8, color="salmon")
    plt.title("Posterior logvar")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_latent_pca(mu: torch.Tensor, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X = mu.detach().cpu().numpy()
    # PCA to 2D
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z2 = X @ Vt[:2].T
    plt.figure(figsize=(6, 6))
    plt.scatter(Z2[:, 0], Z2[:, 1], s=8, alpha=0.6)
    plt.title("Latent posterior (mu) PCA-2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_latent_heatmap(latent_vec: torch.Tensor, out_path: str):
    """Plot a single latent vector as a square-ish heatmap image."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    z = latent_vec.detach().cpu().numpy()
    length = z.shape[-1]
    side = int(np.ceil(np.sqrt(length)))
    pad = side * side - length
    if pad > 0:
        z = np.pad(z, (0, pad), mode="constant")
    img = z.reshape(side, side)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="viridis")
    plt.colorbar()
    plt.title("Latent vector heatmap")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_decoded_image_from_latent(decoder, latent_vec: torch.Tensor, out_path: str):
    """Decode an image from a latent vector using the decoder and save it."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with torch.no_grad():
        z = latent_vec.unsqueeze(0)  # 1 x D
        x = decoder(z).clamp(0, 1).squeeze(0).cpu()  # CxHxW
        img = (x * 255).byte().permute(1, 2, 0).numpy()
    try:
        from PIL import Image
        Image.fromarray(img).save(out_path)
    except Exception as e:
        print(f"Failed to save decoded image from latent: {e}")
