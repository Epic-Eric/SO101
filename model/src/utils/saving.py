import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict

from model.src.interfaces.training import EpochMetrics, TrainingSummary, Checkpoint


def _parse_optional_float(value) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_optional_int(value) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    for conv in (int, lambda x: int(float(x))):
        try:
            return conv(value)
        except (ValueError, TypeError):
            continue
    return None


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
            "kld_raw": (float(m.kld_raw) if getattr(m, "kld_raw", None) is not None else None),
            "one_step_mse": (float(m.one_step_mse) if getattr(m, "one_step_mse", None) is not None else None),
            "rollout_mse": (float(m.rollout_mse) if getattr(m, "rollout_mse", None) is not None else None),
            "latent_drift": (float(m.latent_drift) if getattr(m, "latent_drift", None) is not None else None),
            "val_loss": (m.val_loss if getattr(m, "val_loss", None) is not None else None),
            "val_one_step_mse": (float(m.val_one_step_mse) if getattr(m, "val_one_step_mse", None) is not None else None),
            "beta": (float(m.beta) if getattr(m, "beta", None) is not None else None),
            "rollout_horizon": (int(m.rollout_horizon) if getattr(m, "rollout_horizon", None) is not None else None),
            "gate_threshold": (float(m.gate_threshold) if getattr(m, "gate_threshold", None) is not None else None),
            "contrastive_loss": (float(m.contrastive_loss) if getattr(m, "contrastive_loss", None) is not None else None),
            "action_sensitivity": (float(m.action_sensitivity) if getattr(m, "action_sensitivity", None) is not None else None),
            "latent_action_variance": (float(m.latent_action_variance) if getattr(m, "latent_action_variance", None) is not None else None),
        }
        for m in metrics
    ]
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Also save CSV for convenience
    with open(os.path.join(out_dir, "metrics.csv"), "w") as f:
        header = [
            "epoch",
            "loss",
            "rec_loss",
            "kld",
            "kld_raw",
            "one_step_mse",
            "rollout_mse",
            "latent_drift",
            "val_loss",
            "val_one_step_mse",
            "beta",
            "rollout_horizon",
            "gate_threshold",
            "contrastive_loss",
            "action_sensitivity",
            "latent_action_variance",
        ]
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = [
                r["epoch"],
                r["loss"],
                r["rec_loss"],
                r["kld"],
                "" if r["kld_raw"] is None else r["kld_raw"],
                "" if r["one_step_mse"] is None else r["one_step_mse"],
                "" if r["rollout_mse"] is None else r["rollout_mse"],
                "" if r["latent_drift"] is None else r["latent_drift"],
                "" if r["val_loss"] is None else r["val_loss"],
                "" if r["val_one_step_mse"] is None else r["val_one_step_mse"],
                "" if r["beta"] is None else r["beta"],
                "" if r["rollout_horizon"] is None else r["rollout_horizon"],
                "" if r["gate_threshold"] is None else r["gate_threshold"],
                "" if r["contrastive_loss"] is None else r["contrastive_loss"],
                "" if r["action_sensitivity"] is None else r["action_sensitivity"],
                "" if r["latent_action_variance"] is None else r["latent_action_variance"],
            ]
            f.write(",".join(str(v) for v in vals) + "\n")


def load_metrics(out_dir: str) -> List[EpochMetrics]:
    """Load metrics from metrics.json if present and return list of EpochMetrics."""
    path = os.path.join(out_dir, "metrics.json")
    out: List[EpochMetrics] = []
    if os.path.exists(path):
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
                        val_loss=_parse_optional_float(r.get("val_loss")),
                        kld_raw=_parse_optional_float(r.get("kld_raw")),
                        one_step_mse=_parse_optional_float(r.get("one_step_mse")),
                        rollout_mse=_parse_optional_float(r.get("rollout_mse")),
                        latent_drift=_parse_optional_float(r.get("latent_drift")),
                        val_one_step_mse=_parse_optional_float(r.get("val_one_step_mse")),
                        beta=_parse_optional_float(r.get("beta")),
                        rollout_horizon=_parse_optional_int(r.get("rollout_horizon")),
                        gate_threshold=_parse_optional_float(r.get("gate_threshold")),
                        contrastive_loss=_parse_optional_float(r.get("contrastive_loss")),
                        action_sensitivity=_parse_optional_float(r.get("action_sensitivity")),
                        latent_action_variance=_parse_optional_float(r.get("latent_action_variance")),
                    )
                )
        except Exception:
            return []
        return out

    # Fallback: if metrics.json missing but metrics.csv exists, load CSV to preserve continuity
    csv_path = os.path.join(out_dir, "metrics.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r") as f:
                header = f.readline().strip().split(",")
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 4:
                        continue
                    row = {hdr: (parts[i] if i < len(parts) else "") for i, hdr in enumerate(header)}
                    try:
                        epoch = int(row.get("epoch"))
                        loss = float(row.get("loss"))
                        rec_loss = float(row.get("rec_loss"))
                        kld = float(row.get("kld"))
                        val_loss = _parse_optional_float(row.get("val_loss"))
                    except (TypeError, ValueError):
                        # Skip rows with malformed required numeric fields
                        continue
                    out.append(
                        EpochMetrics(
                            epoch=epoch,
                            loss=loss,
                            rec_loss=rec_loss,
                            kld=kld,
                            val_loss=val_loss,
                            kld_raw=_parse_optional_float(row.get("kld_raw")),
                            one_step_mse=_parse_optional_float(row.get("one_step_mse")),
                            rollout_mse=_parse_optional_float(row.get("rollout_mse")),
                            latent_drift=_parse_optional_float(row.get("latent_drift")),
                            val_one_step_mse=_parse_optional_float(row.get("val_one_step_mse")),
                            beta=_parse_optional_float(row.get("beta")),
                            rollout_horizon=_parse_optional_int(row.get("rollout_horizon")),
                            gate_threshold=_parse_optional_float(row.get("gate_threshold")),
                            contrastive_loss=_parse_optional_float(row.get("contrastive_loss")),
                            action_sensitivity=_parse_optional_float(row.get("action_sensitivity")),
                            latent_action_variance=_parse_optional_float(row.get("latent_action_variance")),
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

    # Write JSON (full rewrite - authoritative)
    rows = [
        {
            "epoch": m.epoch,
            "loss": float(m.loss),
            "rec_loss": float(m.rec_loss),
            "kld": float(m.kld),
            "kld_raw": (float(m.kld_raw) if getattr(m, "kld_raw", None) is not None else None),
            "one_step_mse": (float(m.one_step_mse) if getattr(m, "one_step_mse", None) is not None else None),
            "rollout_mse": (float(m.rollout_mse) if getattr(m, "rollout_mse", None) is not None else None),
            "latent_drift": (float(m.latent_drift) if getattr(m, "latent_drift", None) is not None else None),
            "val_loss": (m.val_loss if getattr(m, "val_loss", None) is not None else None),
            "val_one_step_mse": (float(m.val_one_step_mse) if getattr(m, "val_one_step_mse", None) is not None else None),
            "beta": (float(m.beta) if getattr(m, "beta", None) is not None else None),
            "rollout_horizon": (int(m.rollout_horizon) if getattr(m, "rollout_horizon", None) is not None else None),
            "gate_threshold": (float(m.gate_threshold) if getattr(m, "gate_threshold", None) is not None else None),
            "contrastive_loss": (float(m.contrastive_loss) if getattr(m, "contrastive_loss", None) is not None else None),
            "action_sensitivity": (float(m.action_sensitivity) if getattr(m, "action_sensitivity", None) is not None else None),
            "latent_action_variance": (float(m.latent_action_variance) if getattr(m, "latent_action_variance", None) is not None else None),
        }
        for m in metrics
    ]
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # For CSV, prefer append when adding a new epoch so we don't clobber existing CSV on resume.
    csv_path = os.path.join(out_dir, "metrics.csv")
    header_cols = [
        "epoch",
        "loss",
        "rec_loss",
        "kld",
        "kld_raw",
        "one_step_mse",
        "rollout_mse",
        "latent_drift",
        "val_loss",
        "val_one_step_mse",
        "beta",
        "rollout_horizon",
        "gate_threshold",
        "contrastive_loss",
        "action_sensitivity",
        "latent_action_variance",
    ]
    header = ",".join(header_cols) + "\n"
    if not os.path.exists(csv_path):
        # write full CSV with header
        with open(csv_path, "w") as f:
            f.write(header)
            for r in rows:
                vals = [
                    r["epoch"],
                    r["loss"],
                    r["rec_loss"],
                    r["kld"],
                    "" if r["kld_raw"] is None else r["kld_raw"],
                    "" if r["one_step_mse"] is None else r["one_step_mse"],
                    "" if r["rollout_mse"] is None else r["rollout_mse"],
                    "" if r["latent_drift"] is None else r["latent_drift"],
                    "" if r["val_loss"] is None else r["val_loss"],
                    "" if r["val_one_step_mse"] is None else r["val_one_step_mse"],
                    "" if r["beta"] is None else r["beta"],
                    "" if r["rollout_horizon"] is None else r["rollout_horizon"],
                    "" if r["gate_threshold"] is None else r["gate_threshold"],
                    "" if r["contrastive_loss"] is None else r["contrastive_loss"],
                    "" if r["action_sensitivity"] is None else r["action_sensitivity"],
                    "" if r["latent_action_variance"] is None else r["latent_action_variance"],
                ]
                f.write(",".join(str(v) for v in vals) + "\n")
    else:
        # If we replaced an existing epoch, rewrite full CSV to ensure consistency.
        if replaced:
            with open(csv_path, "w") as f:
                f.write(header)
                for r in rows:
                    vals = [
                        r["epoch"],
                        r["loss"],
                        r["rec_loss"],
                        r["kld"],
                        "" if r["kld_raw"] is None else r["kld_raw"],
                        "" if r["one_step_mse"] is None else r["one_step_mse"],
                        "" if r["rollout_mse"] is None else r["rollout_mse"],
                        "" if r["latent_drift"] is None else r["latent_drift"],
                        "" if r["val_loss"] is None else r["val_loss"],
                        "" if r["val_one_step_mse"] is None else r["val_one_step_mse"],
                        "" if r["beta"] is None else r["beta"],
                        "" if r["rollout_horizon"] is None else r["rollout_horizon"],
                        "" if r["gate_threshold"] is None else r["gate_threshold"],
                        "" if r["contrastive_loss"] is None else r["contrastive_loss"],
                        "" if r["action_sensitivity"] is None else r["action_sensitivity"],
                        "" if r["latent_action_variance"] is None else r["latent_action_variance"],
                    ]
                    f.write(",".join(str(v) for v in vals) + "\n")
        else:
            # append just the new row
            assert rows, "Expected at least one epoch row before appending"
            r = rows[-1]
            with open(csv_path, "a") as f:
                vals = [
                    r["epoch"],
                    r["loss"],
                    r["rec_loss"],
                    r["kld"],
                    "" if r["kld_raw"] is None else r["kld_raw"],
                    "" if r["one_step_mse"] is None else r["one_step_mse"],
                    "" if r["rollout_mse"] is None else r["rollout_mse"],
                    "" if r["latent_drift"] is None else r["latent_drift"],
                    "" if r["val_loss"] is None else r["val_loss"],
                    "" if r["val_one_step_mse"] is None else r["val_one_step_mse"],
                    "" if r["beta"] is None else r["beta"],
                    "" if r["rollout_horizon"] is None else r["rollout_horizon"],
                    "" if r["gate_threshold"] is None else r["gate_threshold"],
                    "" if r["contrastive_loss"] is None else r["contrastive_loss"],
                    "" if r["action_sensitivity"] is None else r["action_sensitivity"],
                    "" if r["latent_action_variance"] is None else r["latent_action_variance"],
                ]
                f.write(",".join(str(v) for v in vals) + "\n")


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
    assert files, "Expected checkpoint list to be non-empty after filtering"
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
