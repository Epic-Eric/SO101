import json
import os
from typing import List, Optional

from model.src.interfaces.training import EpochMetrics, TrainingSummary, Checkpoint


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_epoch_metrics(out_dir: str, metrics: List[EpochMetrics]):
    ensure_dir(out_dir)
    rows = [
        {
            "epoch": m.epoch,
            "loss": float(m.loss),
            "rec_loss": float(m.rec_loss),
            "kld": float(m.kld),
        }
        for m in metrics
    ]
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Also save CSV for convenience
    with open(os.path.join(out_dir, "metrics.csv"), "w") as f:
        f.write("epoch,loss,rec_loss,kld\n")
        for r in rows:
            f.write(f"{r['epoch']},{r['loss']},{r['rec_loss']},{r['kld']}\n")


def save_checkpoint(out_dir: str, ckpt: Checkpoint) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"checkpoint_epoch_{ckpt.epoch}.pt")
    import torch

    payload = {
        "epoch": ckpt.epoch,
        "model_state": ckpt.model_state,
        "optimizer_state": ckpt.optimizer_state,
        "extras": ckpt.extras or {},
    }
    torch.save(payload, path)
    return path


def save_final_model(out_dir: str, model_state) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "model_final.pt")
    import torch

    torch.save(model_state, path)
    return path


def save_training_summary(out_dir: str, summary: TrainingSummary):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "epochs": summary.epochs,
                "final_model_path": summary.final_model_path,
                "checkpoint_paths": summary.checkpoint_paths,
                "metrics": [
                    {
                        "epoch": m.epoch,
                        "loss": m.loss,
                        "rec_loss": m.rec_loss,
                        "kld": m.kld,
                    }
                    for m in summary.metrics
                ],
            },
            f,
            indent=2,
        )
