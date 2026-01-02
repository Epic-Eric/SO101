import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import time
from .dataset import RedBeadDataset
from .bead_model import RedBeadAutoEncoder
from .augmentations import Augmenter


def _resolve_device(device_name: str) -> torch.device:
    if device_name == 'auto':
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)

def train_model(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    device_name: str = 'auto',
    target_hw=(192, 256),
    num_workers: int = 2,
    progress_callback=None,
):
    
    device = _resolve_device(device_name)
    print(f"Using device: {device}")
    if progress_callback is not None:
        try:
            progress_callback({"event": "start", "device": str(device)})
        except Exception:
            pass
    
    # Dataset
    augmenter = Augmenter(prob=0.7)
    full_dataset = RedBeadDataset(data_dir, transform=augmenter, target_hw=target_hw)
    
    if len(full_dataset) == 0:
        print("No data found to train on.")
        return None
        
    # Split (Time based - last 20% for validation)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    indices = list(range(total_size))
    
    train_dataset = Subset(full_dataset, indices[:train_size])
    val_dataset = Subset(full_dataset, indices[train_size:])
    
    # DataLoader performance tuning:
    # - workers speed up cv2.imread + augmentation
    # - persistent_workers avoids respawn each epoch
    # - prefetch overlaps CPU and device
    worker_count = max(0, int(num_workers))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_count,
        persistent_workers=(worker_count > 0),
        prefetch_factor=2 if worker_count > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        persistent_workers=(worker_count > 0),
        prefetch_factor=2 if worker_count > 0 else None,
    )
    
    # Model
    model = RedBeadAutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    model_save_path = os.path.join(data_dir, 'best_model.pth')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        num_batches = max(1, len(train_loader))
        # Throttle UI updates to avoid slowing training too much
        every_n_batches = max(1, num_batches // 20)  # ~20 updates per epoch
        last_progress_ts = 0.0

        last_progress_ts = 0.0
        for batch_i, (images, heatmaps) in enumerate(train_loader, start=1):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            if progress_callback is not None:
                try:
                    now = time.time()
                    if (batch_i % every_n_batches == 0) or (batch_i == 1) or (batch_i == num_batches) or ((now - last_progress_ts) > 0.25):
                        last_progress_ts = now
                        progress_callback({
                            "event": "batch_end",
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "batch": batch_i,
                            "batches": num_batches,
                            "batch_loss": float(loss.item()),
                            "device": str(device),
                        })
                except Exception:
                    pass
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item() * images.size(0)
                
        if len(val_dataset) > 0:
            val_loss /= len(val_dataset)
        else:
            val_loss = 0.0
            
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        if progress_callback is not None:
            try:
                progress_callback({
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "best_val_loss": float(best_val_loss),
                    "saved_best": bool(val_loss < best_val_loss),
                    "device": str(device),
                })
            except Exception:
                pass
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")
            
    if progress_callback is not None:
        try:
            progress_callback({"event": "end", "device": str(device)})
        except Exception:
            pass

    return {"history": history, "device": str(device), "model_path": model_save_path}
