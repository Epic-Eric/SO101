import argparse
import os

import torch

from model.src.core.train_vae import train
from model.src.utils.config import load_yaml_config


def main():
    parser = argparse.ArgumentParser(description="Train a simple VAE on images in a directory and save reconstructions.")
    parser.add_argument("data_dir", type=str, nargs="?", help="Directory containing images (jpg/png)")
    parser.add_argument("out_dir", type=str, nargs="?", help="Output directory for checkpoints and reconstructions")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension")

    args = parser.parse_args()

    # Load defaults from config.yml if present
    cfg_path = os.path.join(os.getcwd(), "config.yml")
    cfg = load_yaml_config(cfg_path)

    data_dir = args.data_dir or cfg.get("data_dir")
    out_dir = args.out_dir or cfg.get("out_dir")
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 10)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 64)
    lr = args.lr if args.lr is not None else cfg.get("lr", 1e-3)
    latent_dim = args.latent_dim if args.latent_dim is not None else cfg.get("latent_dim", 128)

    if not data_dir:
        raise SystemExit("data_dir is required (pass arg or set in config.yml)")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"data_dir not found: {data_dir}")
    if not out_dir:
        raise SystemExit("out_dir is required (pass arg or set in config.yml)")
    os.makedirs(out_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training VAE on data from {data_dir}, saving to {out_dir}")
    print(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, latent_dim={latent_dim}")
    
    final_model_path = train(
        data_dir=data_dir,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        latent_dim=latent_dim,
        device=device,
    )

    print(f"Training complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
