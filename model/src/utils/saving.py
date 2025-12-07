import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict

from model.src.interfaces.training import EpochMetrics, TrainingSummary, Checkpoint


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_artifact_dir(output_dir: str, run_name: Optional[str] = None) -> str:
    """Create a timestamped artifact directory under `output_dir/artifacts/`.

    Returns the full path to the created directory and also writes a basic manifest when called from training.
    """
    base = os.path.join(output_dir, "artifacts")
    ensure_dir(base)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    safe_name = (run_name or "run").replace(" ", "_")
    dir_name = f"{safe_name}_{ts}_{short_id}"
    path = os.path.join(base, dir_name)
    ensure_dir(path)
    return path


def save_manifest(artifact_dir: str, metadata: Dict):
    ensure_dir(artifact_dir)
    with open(os.path.join(artifact_dir, "manifest.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def save_epoch_metrics(out_dir: str, metrics: List[EpochMetrics]):
    ensure_dir(out_dir)
    rows = [
        {
            "epoch": m.epoch,
            "loss": float(m.loss),
            "rec_loss": float(m.rec_loss),
            "kld": float(m.kld),
            "val_loss": (m.val_loss if getattr(m, "val_loss", None) is not None else None),
        }
        for m in metrics
    ]
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Also save CSV for convenience
    with open(os.path.join(out_dir, "metrics.csv"), "w") as f:
        f.write("epoch,loss,rec_loss,kld,val_loss\n")
        for r in rows:
            f.write(f"{r['epoch']},{r['loss']},{r['rec_loss']},{r['kld']},{'' if r['val_loss'] is None else r['val_loss']}\n")


def load_metrics(out_dir: str) -> List[EpochMetrics]:
    """Load metrics from metrics.json if present and return list of EpochMetrics."""
    path = os.path.join(out_dir, "metrics.json")
    out: List[EpochMetrics] = []
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r") as f:
            rows = json.load(f)
        for r in rows:
            out.append(
                EpochMetrics(
                    epoch=int(r.get("epoch")),
                    loss=float(r.get("loss")),
                    rec_loss=float(r.get("rec_loss")),
                    kld=float(r.get("kld")),
                    val_loss=(float(r.get("val_loss")) if r.get("val_loss") is not None else None),
                )
            )
    except Exception:
        return []
    return out


def append_epoch_metric(out_dir: str, metric: EpochMetrics):
    """Append a single EpochMetrics entry to metrics.json and metrics.csv (creates files if missing)."""
    ensure_dir(out_dir)
    metrics = load_metrics(out_dir)
    # Avoid duplicating epoch entries: replace if epoch exists
    replaced = False
    for i, m in enumerate(metrics):
        if m.epoch == metric.epoch:
            metrics[i] = metric
            replaced = True
            break
    if not replaced:
        metrics.append(metric)
    # Sort by epoch
    metrics.sort(key=lambda x: x.epoch)
    # Write JSON
    rows = [
        {
            "epoch": m.epoch,
            "loss": float(m.loss),
            "rec_loss": float(m.rec_loss),
            "kld": float(m.kld),
            "val_loss": (m.val_loss if getattr(m, "val_loss", None) is not None else None),
        }
        for m in metrics
    ]
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)
    # Rewrite CSV
    with open(os.path.join(out_dir, "metrics.csv"), "w") as f:
        f.write("epoch,loss,rec_loss,kld\n")
        for r in rows:
            f.write(f"{r['epoch']},{r['loss']},{r['rec_loss']},{r['kld']}\n")


def save_checkpoint(out_dir: str, ckpt: Checkpoint, metadata: Optional[Dict] = None, filename: Optional[str] = None) -> str:
    """Save a checkpoint payload and return path.

    If `filename` is provided it will be used (and overwritten) inside out_dir;
    otherwise an epoch-based filename is used.
    """
    ensure_dir(out_dir)
    if filename:
        path = os.path.join(out_dir, filename)
    else:
        path = os.path.join(out_dir, f"checkpoint_epoch_{ckpt.epoch}.pt")
    import torch

    payload = {
        "epoch": ckpt.epoch,
        "model_state": ckpt.model_state,
        "optimizer_state": ckpt.optimizer_state,
        "extras": ckpt.extras or {},
    }
    # Save RNG states to allow exact reproducible resume
    try:
        payload["rng_state_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload["rng_state_cuda_all"] = torch.cuda.get_rng_state_all()
    except Exception:
        # If torch not available at save time, skip RNG capture
        pass
    if metadata:
        payload["manifest"] = metadata
    torch.save(payload, path)
    return path


def save_final_model(out_dir: str, model_state, metadata: Optional[Dict] = None) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "model_final.pt")
    import torch

    payload = {"model_state": model_state}
    if metadata:
        payload["manifest"] = metadata
    torch.save(payload, path)
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


def list_runs(output_dir: str) -> List[str]:
    """Return a list of artifact run directories (full paths) under `output_dir/artifacts` that contain a manifest.json."""
    base = os.path.join(output_dir, "artifacts")
    if not os.path.isdir(base):
        return []
    runs = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "manifest.json")):
            runs.append(p)
    return runs


def find_latest_checkpoint(artifact_dir: str) -> Optional[str]:
    """Return path to checkpoint with greatest epoch number in given dir, or None."""
    if not os.path.isdir(artifact_dir):
        return None
    # Prefer explicit latest file if present
    latest_path = os.path.join(artifact_dir, "checkpoint_latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    # Otherwise fallback to epoch-numbered checkpoint files
    files = [f for f in os.listdir(artifact_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not files:
        return None
    def epoch_from_name(n: str):
        try:
            return int(n.replace("checkpoint_epoch_", "").replace(".pt", ""))
        except Exception:
            return -1
    files.sort(key=epoch_from_name)
    return os.path.join(artifact_dir, files[-1])


def load_checkpoint(artifact_dir: str, epoch: Optional[int] = None, expected_model_id: Optional[str] = None) -> Optional[Checkpoint]:
    """Load a checkpoint from artifact_dir. If epoch is None, load latest. If expected_model_id provided, verify against manifest inside checkpoint or manifest.json."""
    import torch

    if epoch is None:
        path = find_latest_checkpoint(artifact_dir)
    else:
        path = os.path.join(artifact_dir, f"checkpoint_epoch_{epoch}.pt")
    if not path or not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu")
    manifest = None
    if isinstance(data, dict):
        manifest = data.get("manifest")
    # also try reading manifest.json
    try:
        with open(os.path.join(artifact_dir, "manifest.json"), "r") as f:
            disc_manifest = json.load(f)
            if not manifest:
                manifest = disc_manifest
    except Exception:
        disc_manifest = None

    if expected_model_id and manifest:
        mid = manifest.get("model_id") if isinstance(manifest, dict) else None
        if mid and mid != expected_model_id:
            raise ValueError(f"Model id mismatch: expected {expected_model_id} but checkpoint has {mid}")

    ckpt = Checkpoint(epoch=data.get("epoch", -1), model_state=data.get("model_state", {}), optimizer_state=data.get("optimizer_state", {}), extras=data.get("extras", {}))
    return ckpt

