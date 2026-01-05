import os
import json
import uuid
import time
from datetime import datetime
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from model.src.interfaces.training import EpochMetrics, Checkpoint
from model.src.interfaces.dataset import ImageJointSequenceDataset
from model.src.utils.normalization import get_default_normalization, denormalize
from model.src.utils.saving import (
    append_epoch_metric,
    load_metrics,
    save_checkpoint,
    save_final_model,
    save_training_summary,
    save_manifest,
    load_checkpoint,
)
from model.src.core.run_context import TrainingRunContext, prepare_run_context
from model.src.models.world_model import WorldModel


def _default_device(device: str | torch.device) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _has_world_model_data(path: str) -> bool:
    """Accept either flat folder with joints.jsonl or episode subfolders."""
    joints_here = os.path.isfile(os.path.join(path, "joints.jsonl"))
    if joints_here:
        return True
    try:
        for name in os.listdir(path):
            cand = os.path.join(path, name)
            if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "joints.jsonl")):
                return True
    except Exception:
        # If directory listing fails, treat as no data.
        pass
    return False


@torch.no_grad()
def _save_debug_images(
    artifact_dir: str,
    epoch: int,
    model: WorldModel,
    batch_images: torch.Tensor,
    batch_actions: torch.Tensor,
    norm_params,
    filename_prefix: str = "",
):
    import torchvision.utils as vutils

    model.eval()
    images = batch_images[:1]
    actions = batch_actions[:1]

    out = model(images, actions)
    # Recon grid: show a longer horizon by sampling skipped frames across the full sequence.
    num_cols = 8
    t_total = int(images.shape[1])
    cols = min(num_cols, t_total)
    idx = torch.linspace(0, t_total - 1, steps=cols, device=images.device).long()
    gt = denormalize(images[0, idx], norm_params).clamp(0, 1)
    rc = denormalize(out.x_rec[0, idx], norm_params).clamp(0, 1)
    grid = torch.cat([gt, rc], dim=0)
    recon_name = f"{filename_prefix}recon_epoch_{epoch:04d}.png"
    vutils.save_image(grid, os.path.join(artifact_dir, recon_name), nrow=cols)

    # Imagine rollout conditioned on actions
    imagined = model.imagine(images[0, 0].unsqueeze(0), actions[0])  # (T,C,H,W)
    im = denormalize(imagined[idx], norm_params).clamp(0, 1)
    grid2 = torch.cat([gt, im], dim=0)
    imagine_name = f"{filename_prefix}imagine_epoch_{epoch:04d}.png"
    vutils.save_image(grid2, os.path.join(artifact_dir, imagine_name), nrow=cols)


def train_world_model(
    data_dir: str,
    out_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 2e-4,
    latent_dim: int = 128,
    deter_dim: int = 256,
    seq_len: int = 16,
    image_size: int = 64,
    action_mode: str = "delta",
    kl_beta: float = 1.0,
    free_nats: float = 0.0,
    action_mask_prob: float = 0.1,
    val_split: float = 0.1,
    device: str | torch.device = "auto",
    log_every: int = 10,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    cache_images: bool = False,
    cache_size: int = 2048,
    preload_images: bool = False,
    preload_dtype: str = "float16",
    amp: bool = True,
    reset_optimizer: bool = False,
    run_context: Optional[TrainingRunContext] = None,
):
    dev = _default_device(device)

    # Accept percent inputs like 10 meaning 10%.
    if val_split is not None and float(val_split) > 1.0:
        if float(val_split) <= 100.0:
            print(f"Interpreting val_split={val_split} as {float(val_split)/100.0:.3f} (percent -> fraction)")
            val_split = float(val_split) / 100.0
        else:
            raise ValueError(f"val_split must be in (0,1) as a fraction, or <=100 as a percent; got {val_split}")

    if dev.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    os.makedirs(out_dir, exist_ok=True)

    if run_context is None:
        run_context = prepare_run_context(
            out_dir=out_dir,
            run_name="world_model",
            load_checkpoint_fn=load_checkpoint,
            prompt_user=False,
        )
    artifact_dir = run_context.artifact_dir
    resume_ckpt = run_context.resume_checkpoint

    manifest_path = os.path.join(artifact_dir, "manifest.json")
    if resume_ckpt is not None and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                model_meta = json.load(f)
            print(f"Resuming run, using existing manifest: {manifest_path}")
        except Exception:
            model_meta = {}
    else:
        model_meta = {}

    model_meta = {
        **model_meta,
        "model_id": model_meta.get("model_id", uuid.uuid4().hex),
        "model_class": "WorldModel",
        "run_name": model_meta.get("run_name", "world_model"),
        "timestamp": model_meta.get("timestamp", datetime.now().isoformat()),
        "data_dir": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "latent_dim": latent_dim,
        "deter_dim": deter_dim,
        "seq_len": seq_len,
        "image_size": image_size,
        "action_mode": action_mode,
        "kl_beta": kl_beta,
        "free_nats": free_nats,
        "action_mask_prob": action_mask_prob,
        "reset_optimizer": bool(reset_optimizer),
    }
    save_manifest(artifact_dir, model_meta)

    print(f"Artifact dir: {artifact_dir}")
    print(f"Using device: {dev}")

    norm = get_default_normalization()
    t0 = time.time()
    print(f"Building dataset from: {data_dir}")
    dataset = ImageJointSequenceDataset(
        root_dir=data_dir,
        seq_len=seq_len,
        image_size=image_size,
        normalize_images=True,
        norm_params=norm,
        action_mode=action_mode,
        normalize_actions=True,
        cache_images=cache_images,
        cache_size=cache_size,
        preload_images=preload_images,
        preload_dtype=preload_dtype,
    )
    print(f"Dataset ready in {time.time() - t0:.1f}s")

    print(f"Loaded world-model dataset: episodes={getattr(dataset, 'num_episodes', 1)} sequences={len(dataset)}")

    model_meta["joint_keys"] = dataset.joint_keys
    model_meta["action_dim"] = dataset.action_dim
    model_meta["num_episodes"] = getattr(dataset, "num_episodes", 1)
    model_meta["sequences"] = len(dataset)
    save_manifest(artifact_dir, model_meta)

    # Train/val split: episode-aware tail split to avoid leakage across episodes.
    val_loader = None
    nw = int(num_workers)
    pm = bool(pin_memory) and (dev.type == "cuda")
    pw = bool(persistent_workers) and nw > 0
    pf = int(prefetch_factor)
    loader_kwargs = {
        "num_workers": nw,
        "pin_memory": pm,
        "persistent_workers": pw,
    }
    # prefetch_factor is only valid when num_workers > 0
    if nw > 0:
        loader_kwargs["prefetch_factor"] = pf

    if val_split and 0.0 < val_split < 1.0:
        val_size = max(1, int(len(dataset) * val_split))
        episode_indices = list(range(getattr(dataset, "num_episodes", 1)))
        val_ep = max(1, int(len(episode_indices) * val_split))
        val_eps = set(episode_indices[-val_ep:])
        windows = getattr(dataset, "windows", None) or []
        train_indices = [i for i, (ep, _) in enumerate(windows) if ep not in val_eps]
        val_indices = [i for i, (ep, _) in enumerate(windows) if ep in val_eps]
        # fallback to size-based split if something went wrong
        if not train_indices or not val_indices:
            indices = list(range(len(dataset)))
            train_indices = indices[:-val_size]
            val_indices = indices[-val_size:]
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
        print(f"Train/Val split: train={len(train_ds)} val={len(val_ds)} (episodes val={len(val_eps)})")
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)

    # World model
    world = WorldModel(
        action_dim=dataset.action_dim,
        latent_dim=latent_dim,
        deter_dim=deter_dim,
        base_channels=64,
        rec_loss="mse",
        output_activation="tanh",
        kl_beta=kl_beta,
        free_nats=free_nats,
    ).to(dev)

    optimizer = torch.optim.Adam(world.parameters(), lr=lr)
    use_amp = bool(amp) and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    all_metrics: List[EpochMetrics] = []
    try:
        existing_metrics = load_metrics(artifact_dir)
        if existing_metrics:
            all_metrics = existing_metrics
    except Exception:
        all_metrics = []

    start_epoch = 1
    if resume_ckpt is not None:
        try:
            print(f"Resuming from checkpoint epoch {resume_ckpt.epoch}")
            world.load_state_dict(resume_ckpt.model_state)
            if bool(reset_optimizer):
                print("Resetting optimizer state (weights-only resume)")
            else:
                optimizer.load_state_dict(resume_ckpt.optimizer_state)
            start_epoch = resume_ckpt.epoch + 1
        except Exception as e:
            print(f"Failed to load resume checkpoint state: {e}")

    best_loss = float("inf")
    best_path = None

    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epochs", unit="epoch"):
        world.train()
        running_loss = 0.0
        running_rec = 0.0
        running_kld = 0.0
        n_batches = 0

        # Keep one batch for debug images (varies per-epoch due to shuffle).
        debug_images = None
        debug_actions = None
        val_debug_images = None
        val_debug_actions = None

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step_idx, (images, actions) in enumerate(pbar, start=1):
            images = images.to(dev, non_blocking=True)
            actions = actions.to(dev, non_blocking=True)

            if debug_images is None:
                debug_images = images[:1].detach()
                debug_actions = actions[:1].detach()

            if action_mask_prob and action_mask_prob > 0:
                mask = torch.rand(actions.shape[0], actions.shape[1], 1, device=actions.device) < float(
                    action_mask_prob
                )
                actions = actions.masked_fill(mask, 0.0)

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = world(images, actions)
                scaler.scale(out.loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = world(images, actions)
                out.loss.backward()
                optimizer.step()

            running_loss += float(out.loss.detach().cpu())
            running_rec += float(out.rec_loss.detach().cpu())
            running_kld += float(out.kld.detach().cpu())
            n_batches += 1

            if log_every and (step_idx % max(1, int(log_every)) == 0 or step_idx == 1):
                pbar.set_postfix(
                    {
                        "loss": f"{(running_loss / n_batches):.3f}",
                        "rec": f"{(running_rec / n_batches):.3f}",
                        "kld": f"{(running_kld / n_batches):.3f}",
                    }
                )

        epoch_loss = running_loss / max(1, n_batches)
        epoch_rec = running_rec / max(1, n_batches)
        epoch_kld = running_kld / max(1, n_batches)

        # Validation
        val_loss = None
        if val_loader is not None:
            world.eval()
            v_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for v_images, v_actions in val_loader:
                    v_images = v_images.to(dev, non_blocking=True)
                    v_actions = v_actions.to(dev, non_blocking=True)

                    if val_debug_images is None:
                        val_debug_images = v_images[:1].detach()
                        val_debug_actions = v_actions[:1].detach()

                    v_out = world(v_images, v_actions)
                    v_loss += float(v_out.loss.detach().cpu())
                    v_batches += 1
            val_loss = v_loss / max(1, v_batches)

        metric = EpochMetrics(epoch=epoch, loss=epoch_loss, rec_loss=epoch_rec, kld=epoch_kld, val_loss=val_loss)
        all_metrics.append(metric)
        append_epoch_metric(artifact_dir, metric)

        save_checkpoint(
            artifact_dir,
            Checkpoint(epoch=epoch, model_state=world.state_dict(), optimizer_state=optimizer.state_dict(), extras={}),
            metadata=model_meta,
            filename="checkpoint_latest.pt",
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = save_checkpoint(
                artifact_dir,
                Checkpoint(epoch=epoch, model_state=world.state_dict(), optimizer_state=optimizer.state_dict(), extras={}),
                metadata=model_meta,
                filename="checkpoint_best.pt",
            )

        # Save a couple of debug images
        try:
            if debug_images is not None and debug_actions is not None:
                _save_debug_images(artifact_dir, epoch, world, debug_images, debug_actions, dataset.norm_params)

            if val_debug_images is not None and val_debug_actions is not None:
                _save_debug_images(
                    artifact_dir,
                    epoch,
                    world,
                    val_debug_images,
                    val_debug_actions,
                    dataset.norm_params,
                    filename_prefix="val_",
                )
        except Exception as e:
            print(f"Could not save debug images: {e}")

        print(
            f"Epoch {epoch}/{epochs} | loss={epoch_loss:.4f} rec={epoch_rec:.4f} kld={epoch_kld:.4f}"
            + (f" val={val_loss:.4f}" if val_loss is not None else "")
        )

    final_model_path = save_final_model(artifact_dir, world.state_dict(), metadata=model_meta)

    # Training summary (reusing TrainingSummary schema)
    from model.src.interfaces.training import TrainingSummary

    summary = TrainingSummary(
        epochs=epochs,
        metrics=all_metrics,
        final_model_path=final_model_path,
        checkpoint_paths=[os.path.join(artifact_dir, "checkpoint_latest.pt")] + ([best_path] if best_path else []),
    )
    save_training_summary(artifact_dir, summary)
    return final_model_path
