import os
import json
import uuid
from datetime import datetime
from typing import Tuple, List

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset

from model.src.models.vae import VAE
from model.src.interfaces.training import EpochMetrics, TrainingSummary, Checkpoint
from model.src.interfaces.dataset import ImageFolder64Dataset
from model.src.utils.normalization import get_default_normalization, denormalize
from model.src.utils.saving import (
    save_epoch_metrics,
    append_epoch_metric,
    load_metrics,
    save_checkpoint,
    save_final_model,
    save_training_summary,
    make_artifact_dir,
    save_manifest,
    list_runs,
    load_checkpoint,
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
    val_split: float = 0.1,
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

    # If there are previous runs under out_dir/artifacts, offer to continue or rewrite
    existing = list_runs(out_dir)
    artifact_dir = None
    resume_ckpt = None
    if existing:
        print("Found existing runs in output artifacts:")
        runs_meta = []
        for i, p in enumerate(existing):
            try:
                with open(os.path.join(p, "manifest.json"), "r") as f:
                    m = json.load(f)
            except Exception:
                m = {"path": p}
            runs_meta.append((p, m))
            label = m.get("run_name") if isinstance(m, dict) else p
            mid = m.get("model_id") if isinstance(m, dict) else None
            ts = m.get("timestamp") if isinstance(m, dict) else None
            print(f"[{i}] {os.path.basename(p)} run_name={label} model_id={mid} ts={ts}")

        choice = input("Choose '[c] <index>' to continue, '[r] <index>' to rewrite (start new run), or 'n' for new run [n]: ").strip()
        if choice.startswith("c"):
            try:
                idx = int(choice.split()[1])
                artifact_dir = existing[idx]
                # Attempt to load latest checkpoint
                resume_ckpt = load_checkpoint(artifact_dir)
                if resume_ckpt is None:
                    print("No checkpoint found to resume; starting fresh in a new artifact dir.")
                    artifact_dir = None
            except Exception as e:
                print(f"Could not parse choice/continue: {e}; starting new run")
                artifact_dir = None
        elif choice.startswith("r"):
            try:
                idx = int(choice.split()[1])
                # create a new artifact dir preserving run name if present
                try:
                    with open(os.path.join(existing[idx], "manifest.json"), "r") as f:
                        oldm = json.load(f)
                        run_name = oldm.get("run_name")
                except Exception:
                    run_name = None
                artifact_dir = make_artifact_dir(out_dir, run_name=run_name)
            except Exception as e:
                print(f"Could not parse rewrite choice: {e}; creating a new run")
                artifact_dir = None
        else:
            artifact_dir = None

    if artifact_dir is None:
        # create a fresh artifact dir for this run
        artifact_dir = make_artifact_dir(out_dir, run_name="vae")

    # Write or reuse run manifest / metadata. If resuming, preserve original manifest values
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    if resume_ckpt is not None and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                model_meta = json.load(f)
            print(f"Resuming run, using existing manifest: {manifest_path}")
        except Exception:
            model_meta = {
                "model_id": uuid.uuid4().hex,
                "model_class": VAE.__name__,
                "run_name": "vae",
                "timestamp": datetime.now().isoformat(),
                "latent_dim": latent_dim,
                "lr": lr,
                "batch_size": batch_size,
            }
            save_manifest(artifact_dir, model_meta)
    else:
        model_meta = {
            "model_id": uuid.uuid4().hex,
            "model_class": VAE.__name__,
            "run_name": "vae",
            "timestamp": datetime.now().isoformat(),
            "latent_dim": latent_dim,
            "lr": lr,
            "batch_size": batch_size,
        }
        save_manifest(artifact_dir, model_meta)

    print(f"Artifact dir: {artifact_dir}")

    dataset = ImageFolder64Dataset(data_dir, normalize=True, norm_params=get_default_normalization())
    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")

    # helper to get dataset length in a type-safe way for linters that
    # can't detect __len__ on Dataset objects
    def _safe_len(ds) -> int:
        try:
            return int(len(ds))
        except Exception:
            # fallback to iterating (only used if len is not implemented)
            ct = 0
            for _ in ds:
                ct += 1
            return ct

    # Create deterministic train/val split (use tail of dataset for validation so it is reproducible)
    val_loader = None
    train_loader = None
    if val_split and 0.0 < val_split < 1.0:
        val_size = max(1, int(len(dataset) * val_split))
        indices = list(range(len(dataset)))
        train_indices = indices[:-val_size]
        val_indices = indices[-val_size:]
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Train/Val split: train={len(train_ds)} val={len(val_ds)}")
        # annotate manifest metadata but do not overwrite on resume
        model_meta["val_split"] = val_split
        model_meta["val_size"] = val_size
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Choose activation/loss based on normalization
    use_norm = True
    output_activation = "tanh" if use_norm else "sigmoid"
    rec_loss = "mse" if use_norm else "bce"
    # Build model and optimizer
    vae = VAE(in_channels=3, latent_dim=latent_dim, output_activation=output_activation, rec_loss=rec_loss).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Load existing metrics if present (resume or continuing run)
    all_metrics: List[EpochMetrics] = []
    try:
        existing_metrics = load_metrics(artifact_dir)
        if existing_metrics:
            all_metrics = existing_metrics
    except Exception:
        all_metrics = []
    checkpoint_paths: List[str] = []

    start_epoch = 1
    if resume_ckpt is not None:
        try:
            print(f"Resuming from checkpoint epoch {resume_ckpt.epoch}")
            vae.load_state_dict(resume_ckpt.model_state)
            optimizer.load_state_dict(resume_ckpt.optimizer_state)
            start_epoch = resume_ckpt.epoch + 1
        except Exception as e:
            print(f"Failed to load resume checkpoint state: {e}")

    # Determine current best loss (if resuming, try to use existing summary.json)
    best_loss = float("inf")
    summary_path = os.path.join(artifact_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                prev = json.load(f)
                prev_metrics = prev.get("metrics", [])
                for m in prev_metrics:
                    if isinstance(m, dict) and "loss" in m:
                        best_loss = min(best_loss, float(m["loss"]))
        except Exception:
            best_loss = float("inf")
    best_path = None
    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epochs", unit="epoch"):
        vae.train()
        running_loss = 0.0
        running_rec = 0.0
        running_kld = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            x_rec, mu, logvar = vae(batch)
            loss, rec_l, kld = vae.loss_fn(batch, x_rec, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.size(0)
            running_rec += rec_l.item() * batch.size(0)
            running_kld += kld.item() * batch.size(0)

        train_n = _safe_len(train_loader.dataset)
        epoch_loss = running_loss / train_n if train_n > 0 else float("nan")
        epoch_rec = running_rec / train_n if train_n > 0 else float("nan")
        epoch_kld = running_kld / train_n if train_n > 0 else float("nan")
        print(f"Epoch {epoch}/{epochs} | loss={epoch_loss:.4f} rec={epoch_rec:.4f} kld={epoch_kld:.4f}")

        # Compute validation loss if available
        val_loss = None
        if val_loader is not None:
            vae.eval()
            v_running_loss = 0.0
            v_running_rec = 0.0
            v_running_kld = 0.0
            with torch.no_grad():
                for vb in val_loader:
                    vb = vb.to(device)
                    v_rec, v_mu, v_logvar = vae(vb)
                    v_loss, v_rec_l, v_kld = vae.loss_fn(vb, v_rec, v_mu, v_logvar)
                    v_running_loss += v_loss.item() * vb.size(0)
                    v_running_rec += v_rec_l.item() * vb.size(0)
                    v_running_kld += v_kld.item() * vb.size(0)
            val_n = _safe_len(val_loader.dataset)
            if val_n > 0:
                val_loss = v_running_loss / val_n

        metric = EpochMetrics(epoch=epoch, loss=epoch_loss, rec_loss=epoch_rec, kld=epoch_kld, val_loss=val_loss)
        all_metrics.append(metric)
        # Append metric to continuous log (ensures resume will pick up)
        append_epoch_metric(artifact_dir, metric)

        # Save checkpoint (include manifest metadata in payload)
        # Only keep latest and best checkpoints to reduce disk usage.
        latest_path = save_checkpoint(
            artifact_dir,
            Checkpoint(epoch=epoch, model_state=vae.state_dict(), optimizer_state=optimizer.state_dict(), extras={"latent_dim": latent_dim}),
            metadata=model_meta,
            filename="checkpoint_latest.pt",
        )
        # Track latest
        latest = latest_path
        # Check for improvement and save best if improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = save_checkpoint(
                artifact_dir,
                Checkpoint(epoch=epoch, model_state=vae.state_dict(), optimizer_state=optimizer.state_dict(), extras={"latent_dim": latent_dim}),
                metadata=model_meta,
                filename="checkpoint_best.pt",
            )

        # Save a sample reconstruction grid
        vae.eval()
        with torch.no_grad():
            sample = next(iter(train_loader))[:256].to(device)
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
                recon_dir = os.path.join(artifact_dir, "recon")
                os.makedirs(recon_dir, exist_ok=True)
                recon_path = os.path.join(recon_dir, f"recon_epoch_{epoch}.png")
                print(f"Saving recon image to: {recon_path}")
                Image.fromarray(grid_img).save(recon_path)
            except Exception as e:
                print(f"Failed to save recon image: {e}")

            # Latent posterior visualization
            try:
                # Save each visualization into its own subdirectory under the run artifact dir
                hist_dir = os.path.join(artifact_dir, "latent_hist")
                pca_dir = os.path.join(artifact_dir, "latent_pca")
                heatmap_dir = os.path.join(artifact_dir, "latent_heatmap")
                decoded_dir = os.path.join(artifact_dir, "latent_decoded")
                for d in (hist_dir, pca_dir, heatmap_dir, decoded_dir):
                    os.makedirs(d, exist_ok=True)

                hist_path = os.path.join(hist_dir, f"latent_hist_epoch_{epoch}.png")
                pca_path = os.path.join(pca_dir, f"latent_pca_epoch_{epoch}.png")
                heatmap_path = os.path.join(heatmap_dir, f"latent_heatmap_epoch_{epoch}.png")
                decoded_path = os.path.join(decoded_dir, f"latent_decoded_epoch_{epoch}.png")

                plot_latent_hist(mu, logvar, hist_path)
                plot_latent_pca(mu, pca_path)
                # Sample latent vector visualizations
                plot_latent_heatmap(mu[0], heatmap_path)
                save_decoded_image_from_latent(vae.decoder, mu[0], decoded_path)
            except Exception as e:
                print(f"Failed to plot latent visuals: {e}")

    # Final save
    final_path = save_final_model(artifact_dir, vae.state_dict(), metadata=model_meta)

    # If a best model was found, ensure it is recorded in checkpoint_paths
    cp_paths = []
    latest_file = os.path.join(artifact_dir, "checkpoint_latest.pt")
    if os.path.exists(latest_file):
        cp_paths.append(latest_file)
    best_file = os.path.join(artifact_dir, "checkpoint_best.pt")
    if os.path.exists(best_file) and best_file != latest_file:
        cp_paths.append(best_file)

    save_training_summary(artifact_dir, TrainingSummary(epochs=epochs, metrics=all_metrics, final_model_path=final_path, checkpoint_paths=cp_paths))
    return final_path
