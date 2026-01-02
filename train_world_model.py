import argparse
import os

import torch

from model.src.core.train_world_model import _has_world_model_data, train_world_model
from model.src.utils.config import load_yaml_config


def _pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train a VAE+RSSM world model from image sequences + joints.jsonl")
    parser.add_argument("data_dir", type=str, nargs="?", help="Directory containing joints.jsonl and frames (e.g. data/captured_images_and_joints)")
    parser.add_argument("out_dir", type=str, nargs="?", help="Output directory (e.g. output/)")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--deter_dim", type=int, default=None)

    parser.add_argument("--action_mode", type=str, default=None, choices=["delta", "pos"], help="How to convert joints into action vectors")
    parser.add_argument("--kl_beta", type=float, default=None)
    parser.add_argument("--free_nats", type=float, default=None)
    parser.add_argument("--val_split", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None, help="Update per-batch loss stats every N steps")

    # Performance knobs (useful for Colab / Google Drive)
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers; 0 can be faster on Google Drive")
    parser.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch factor (only if num_workers>0)")
    parser.add_argument("--no_persistent_workers", action="store_true", help="Disable persistent workers")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP mixed precision")
    parser.add_argument("--cache_images", action="store_true", help="Cache transformed frames in RAM (speeds up slow disk)")
    parser.add_argument("--cache_size", type=int, default=None, help="Max number of frames to keep in RAM cache")
    parser.add_argument(
        "--preload_images",
        action="store_true",
        help="Preload ALL transformed frames into RAM once (fastest; usually use --num_workers 0)",
    )
    parser.add_argument(
        "--preload_dtype",
        type=str,
        default=None,
        choices=["float16", "float32"],
        help="Storage dtype for preloaded frames (default: float16)",
    )

    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")

    args = parser.parse_args()

    cfg_path = os.path.join(os.getcwd(), "config.yml")
    cfg = load_yaml_config(cfg_path)

    # Prefer dedicated keys, then fallback if config already points at the right folder.
    cfg_data_dir = cfg.get("world_data_dir")
    if not cfg_data_dir:
        cand = cfg.get("data_dir")
        if isinstance(cand, str) and os.path.isdir(cand) and _has_world_model_data(cand):
            cfg_data_dir = cand

    data_dir = args.data_dir or cfg_data_dir
    out_dir = args.out_dir or cfg.get("world_out_dir") or cfg.get("out_dir")

    if not data_dir:
        raise SystemExit("data_dir is required (pass arg or set world_data_dir in config.yml)")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"data_dir not found: {data_dir}")
    if not _has_world_model_data(data_dir):
        raise SystemExit(
            f"Expected joints.jsonl and frame images in {data_dir} or in at least one immediate subdirectory"
        )

    if not out_dir:
        raise SystemExit("out_dir is required (pass arg or set world_out_dir/out_dir in config.yml)")
    os.makedirs(out_dir, exist_ok=True)

    epochs = args.epochs if args.epochs is not None else int(cfg.get("world_epochs", cfg.get("epochs", 100)))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("world_batch_size", cfg.get("batch_size", 16)))
    lr = args.lr if args.lr is not None else float(cfg.get("world_lr", cfg.get("lr", 2e-4)))

    seq_len = args.seq_len if args.seq_len is not None else int(cfg.get("seq_len", 16))
    image_size = args.image_size if args.image_size is not None else int(cfg.get("image_size", 64))
    latent_dim = args.latent_dim if args.latent_dim is not None else int(cfg.get("world_latent_dim", cfg.get("latent_dim", 128)))
    deter_dim = args.deter_dim if args.deter_dim is not None else int(cfg.get("deter_dim", 256))

    action_mode = args.action_mode if args.action_mode is not None else str(cfg.get("action_mode", "delta"))
    kl_beta = args.kl_beta if args.kl_beta is not None else float(cfg.get("kl_beta", 1.0))
    free_nats = args.free_nats if args.free_nats is not None else float(cfg.get("free_nats", 0.0))
    val_split = args.val_split if args.val_split is not None else float(cfg.get("val_split", 0.1))
    log_every = args.log_every if args.log_every is not None else int(cfg.get("world_log_every", cfg.get("log_every", 10)))

    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("world_num_workers", 2))
    prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else int(cfg.get("world_prefetch_factor", 2))
    persistent_workers = not bool(args.no_persistent_workers)
    amp = not bool(args.no_amp)
    cache_images = bool(args.cache_images)
    cache_size = args.cache_size if args.cache_size is not None else int(cfg.get("world_cache_size", 2048))
    preload_images = bool(args.preload_images)
    preload_dtype = args.preload_dtype if args.preload_dtype is not None else str(cfg.get("world_preload_dtype", "float16"))

    if preload_images and num_workers != 0:
        print("Note: --preload_images duplicates memory across DataLoader workers; consider --num_workers 0")

    device = args.device
    if device == "auto":
        device = _pick_device()

    print(f"Using device: {device}")
    print(f"Training world model on {data_dir}, saving to {out_dir}")
    print(f"epochs={epochs} batch_size={batch_size} lr={lr} seq_len={seq_len} latent_dim={latent_dim} deter_dim={deter_dim} action_mode={action_mode}")

    final_model_path = train_world_model(
        data_dir=data_dir,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        latent_dim=latent_dim,
        deter_dim=deter_dim,
        seq_len=seq_len,
        image_size=image_size,
        action_mode=action_mode,
        kl_beta=kl_beta,
        free_nats=free_nats,
        val_split=val_split,
        device=device,
        log_every=log_every,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        cache_images=cache_images,
        cache_size=cache_size,
        preload_images=preload_images,
        preload_dtype=preload_dtype,
        amp=amp,
    )

    print(f"Training complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
