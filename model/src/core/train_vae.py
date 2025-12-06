import os
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader

from model.src.models.vae import VAE
from model.src.interfaces.training import EpochMetrics, TrainingSummary, Checkpoint
from model.src.interfaces.dataset import ImageFolder64Dataset
from model.src.utils.normalization import get_default_normalization, denormalize
from model.src.utils.saving import (
    save_epoch_metrics,
    save_checkpoint,
    save_final_model,
    save_training_summary,
)
from model.visualization.latent import plot_latent_hist, plot_latent_pca, plot_latent_heatmap, save_decoded_image_from_latent


# moved to interface


def train(
    data_dir: str,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    latent_dim: int = 128,
    device: str | torch.device = "auto",
):
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    else:
        try:
            device = torch.device(device)
            print(f"Using device: {device}")
        except Exception:
            raise ValueError(f"Invalid device: {device}")

    os.makedirs(out_dir, exist_ok=True)

    dataset = ImageFolder64Dataset(data_dir, normalize=True, norm_params=get_default_normalization())
    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Choose activation/loss based on normalization
    use_norm = True
    output_activation = "tanh" if use_norm else "sigmoid"
    rec_loss = "mse" if use_norm else "bce"
    vae = VAE(in_channels=3, latent_dim=latent_dim, output_activation=output_activation, rec_loss=rec_loss).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    all_metrics: List[EpochMetrics] = []
    checkpoint_paths: List[str] = []

    for epoch in range(1, epochs + 1):
        vae.train()
        running_loss = 0.0
        running_rec = 0.0
        running_kld = 0.0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_rec, mu, logvar = vae(batch)
            loss, rec_l, kld = vae.loss_fn(batch, x_rec, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.size(0)
            running_rec += rec_l.item() * batch.size(0)
            running_kld += kld.item() * batch.size(0)

        epoch_loss = running_loss / len(dataset)
        epoch_rec = running_rec / len(dataset)
        epoch_kld = running_kld / len(dataset)
        print(f"Epoch {epoch}/{epochs} | loss={epoch_loss:.4f} rec={epoch_rec:.4f} kld={epoch_kld:.4f}")

        all_metrics.append(EpochMetrics(epoch=epoch, loss=epoch_loss, rec_loss=epoch_rec, kld=epoch_kld))
        save_epoch_metrics(out_dir, all_metrics)

        # Save checkpoint
        ckpt_path = save_checkpoint(out_dir, Checkpoint(
            epoch=epoch,
            model_state=vae.state_dict(),
            optimizer_state=optimizer.state_dict(),
            extras={"latent_dim": latent_dim},
        ))
        checkpoint_paths.append(ckpt_path)

        # Save a sample reconstruction grid
        vae.eval()
        with torch.no_grad():
            sample = next(iter(loader))[:256].to(device)
            rec, mu, logvar = vae(sample)
            # Denormalize to visualize
            sample_vis = denormalize(sample[:16], dataset.norm_params).clamp(0, 1)
            rec_vis = denormalize(rec[:16], dataset.norm_params).clamp(0, 1)
            grid = torch.cat([sample_vis, rec_vis], dim=0)  # 32 images
            # Arrange manually into a single image (4x8)
            def make_grid(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
                b, c, h, w = t.size()
                assert b == rows * cols
                grid = (
                    t.view(rows, cols, c, h, w)
                    .permute(2, 0, 3, 1, 4)
                    .contiguous()
                    .view(c, rows * h, cols * w)
                )
                return grid

            rows, cols = 4, 8
            grid_img = make_grid(grid, rows, cols)
            grid_img = (grid_img.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
            try:
                from PIL import Image
                Image.fromarray(grid_img).save(os.path.join(out_dir, f"recon_epoch_{epoch}.png"))
            except Exception as e:
                print(f"Failed to save recon image: {e}")

            # Latent posterior visualization
            try:
                plot_latent_hist(mu, logvar, os.path.join(out_dir, f"latent_hist_epoch_{epoch}.png"))
                plot_latent_pca(mu, os.path.join(out_dir, f"latent_pca_epoch_{epoch}.png"))
                # Sample latent vector visualizations
                plot_latent_heatmap(mu[0], os.path.join(out_dir, f"latent_heatmap_epoch_{epoch}.png"))
                save_decoded_image_from_latent(vae.decoder, mu[0], os.path.join(out_dir, f"latent_decoded_epoch_{epoch}.png"))
            except Exception as e:
                print(f"Failed to plot latent visuals: {e}")

    # Final save
    final_path = save_final_model(out_dir, vae.state_dict())

    save_training_summary(out_dir, TrainingSummary(
        epochs=epochs,
        metrics=all_metrics,
        final_model_path=final_path,
        checkpoint_paths=checkpoint_paths,
    ))
    return final_path
