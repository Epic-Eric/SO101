import argparse
import os

import torch

from model.src.core.run_context import prepare_run_context
from model.src.core.train_world_model import train_world_model
from model.src.utils.config import load_yaml_config
from model.src.utils.saving import load_checkpoint


def _pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _has_world_model_data(path: str) -> bool:
    """Accept either a flat folder with joints.jsonl or episode subfolders.

    For episode roots, we accept any nested folder containing joints.jsonl.
    """

    if os.path.isfile(os.path.join(path, "joints.jsonl")):
        return True
    try:
        for dirpath, _, filenames in os.walk(path):
            if "joints.jsonl" in filenames:
                return True
    except Exception:
        return False
    return False


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
    parser.add_argument("--kl_warmup_epochs", type=int, default=None, help="Warm-up epochs for KL beta (linear 0->beta)")
    parser.add_argument("--free_nats", type=float, default=None)
    parser.add_argument(
        "--val_split",
        type=float,
        default=None,
        help="Validation split as fraction (e.g. 0.1) or percent (e.g. 10 for 10%)",
    )
    parser.add_argument("--log_every", type=int, default=None, help="Update per-batch loss stats every N steps")

    # Performance knobs (useful for Colab / Google Drive)
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers; 0 can be faster on Google Drive")
    parser.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch factor (only if num_workers>0)")
    parser.add_argument("--no_persistent_workers", action="store_true", help="Disable persistent workers")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP mixed precision")
    parser.add_argument("--cache_images", action="store_true", help="Cache transformed frames in RAM (speeds up slow disk)")
    parser.add_argument("--cache_size", type=int, default=None, help="Max number of frames to keep in RAM cache")
    parser.add_argument(
        "--action_mask_prob",
        type=float,
        default=None,
        help="Probability of masking actions during training for autoregressive robustness",
    )
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
    parser.add_argument("--rssm_gate_threshold", type=float, default=None, help="1-step latent MSE threshold for gating RSSM->encoder grads")
    parser.add_argument("--rollout_horizon", type=int, default=None, help="Short latent rollout horizon for metrics (e.g., 3-5)")
    parser.add_argument(
        "--no_prompt",
        action="store_true",
        help="Do not prompt for artifact resume/rewrite; always start a new run (recommended for batch jobs)",
    )

    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="When resuming from a checkpoint, load model weights but reset optimizer state",
    )

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
        raise SystemExit(f"Expected joints.jsonl in {data_dir} or in at least one immediate subdirectory")

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
    kl_warmup_epochs = args.kl_warmup_epochs if args.kl_warmup_epochs is not None else cfg.get("kl_warmup_epochs")
    if kl_warmup_epochs is not None:
        kl_warmup_epochs = int(kl_warmup_epochs)
    free_nats = args.free_nats if args.free_nats is not None else float(cfg.get("free_nats", 0.0))
    rssm_gate_threshold = (
        args.rssm_gate_threshold if args.rssm_gate_threshold is not None else float(cfg.get("rssm_gate_threshold", 0.25))
    )
    rollout_horizon = args.rollout_horizon if args.rollout_horizon is not None else int(cfg.get("rollout_horizon", cfg.get("short_rollout_horizon", 3)))
    val_split = args.val_split if args.val_split is not None else float(cfg.get("val_split", 0.1))
    # Accept percent inputs like 10 meaning 10%.
    if val_split is not None and val_split > 1.0:
        if val_split <= 100.0:
            print(f"Interpreting val_split={val_split} as {val_split/100.0:.3f} (percent -> fraction)")
            val_split = float(val_split) / 100.0
        else:
            raise SystemExit(f"val_split must be in (0,1) as a fraction, or <=100 as a percent; got {val_split}")
    log_every = args.log_every if args.log_every is not None else int(cfg.get("world_log_every", cfg.get("log_every", 10)))

    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("world_num_workers", 2))
    prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else int(cfg.get("world_prefetch_factor", 2))
    persistent_workers = not bool(args.no_persistent_workers)
    amp = not bool(args.no_amp)
    cache_images = bool(args.cache_images)
    cache_size = args.cache_size if args.cache_size is not None else int(cfg.get("world_cache_size", 2048))
    preload_images = bool(args.preload_images)
    preload_dtype = args.preload_dtype if args.preload_dtype is not None else str(cfg.get("world_preload_dtype", "float16"))
    action_mask_prob = args.action_mask_prob if args.action_mask_prob is not None else float(cfg.get("world_action_mask_prob", 0.1))

    if preload_images and num_workers != 0:
        print("Note: --preload_images duplicates memory across DataLoader workers; consider --num_workers 0")

    device = args.device
    if device == "auto":
        device = _pick_device()

    print(f"Using device: {device}")
    print(f"Training world model on {data_dir}, saving to {out_dir}")
    print(
        f"epochs={epochs} batch_size={batch_size} lr={lr} seq_len={seq_len} latent_dim={latent_dim} "
        f"deter_dim={deter_dim} action_mode={action_mode} action_mask_prob={action_mask_prob}"
    )

    prompt_user = not args.no_prompt
    run_context = prepare_run_context(
        out_dir=out_dir,
        run_name="world_model",
        load_checkpoint_fn=load_checkpoint,
        prompt_user=prompt_user,
    )

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
        kl_warmup_epochs=kl_warmup_epochs,
        free_nats=free_nats,
        rssm_gate_threshold=rssm_gate_threshold,
        short_roll_horizon=rollout_horizon,
        action_mask_prob=action_mask_prob,
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
        reset_optimizer=bool(args.reset_optimizer),
        run_context=run_context,
    )

    print(f"Training complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
