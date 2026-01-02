import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

from model.src.interfaces.training import Checkpoint
from model.src.utils.saving import list_runs, make_artifact_dir


@dataclass
class TrainingRunContext:
    """Container describing where to store artifacts and whether to resume from a checkpoint."""

    artifact_dir: str
    resume_checkpoint: Optional[Checkpoint] = None


def _default_manifest_loader(path: str) -> Optional[dict]:
    try:
        with open(os.path.join(path, "manifest.json"), "r") as file:
            return json.load(file)
    except Exception:
        return None


def _read_run_label(manifest_loader: Callable[[str], Optional[dict]], path: str) -> str | None:
    loader = manifest_loader or _default_manifest_loader
    try:
        meta = loader(path)
    except Exception:
        meta = None
    if isinstance(meta, dict):
        return meta.get("run_name")
    return None


def prepare_run_context(
    out_dir: str,
    run_name: str,
    load_checkpoint_fn: Callable[[str], Optional[Checkpoint]],
    manifest_loader: Callable[[str], Optional[dict]] | None = None,
    prompt_user: bool = True,
) -> TrainingRunContext:
    """
    Select or create an artifact directory and (optionally) a resume checkpoint.

    This keeps orchestration (prompting/selection) out of core training loops.
    """
    existing = list_runs(out_dir)
    artifact_dir = None
    resume_ckpt: Optional[Checkpoint] = None

    if existing and prompt_user:
        print("Found existing runs in output artifacts:")
        manifest_reader = manifest_loader or _default_manifest_loader
        for i, p in enumerate(existing):
            meta = manifest_reader(p)
            label = meta.get("run_name") if isinstance(meta, dict) else p
            mid = meta.get("model_id") if isinstance(meta, dict) else None
            ts = meta.get("timestamp") if isinstance(meta, dict) else None
            print(f"[{i}] {os.path.basename(p)} run_name={label} model_id={mid} ts={ts}")

        choice = input(
            "Choose '[c] <index>' to continue, '[r] <index>' to rewrite (start new run), or 'n' for new run [n]: "
        ).strip()
        if choice.startswith("c"):
            try:
                idx = int(choice.split()[1])
                artifact_dir = existing[idx]
                resume_ckpt = load_checkpoint_fn(artifact_dir)
                if resume_ckpt is None:
                    print("No checkpoint found to resume; starting fresh in a new artifact dir.")
                    artifact_dir = None
            except Exception as exc:
                print(f"Could not parse choice/continue: {exc}; starting new run")
                artifact_dir = None
        elif choice.startswith("r"):
            try:
                idx = int(choice.split()[1])
                preserved_name = _read_run_label(manifest_loader or _default_manifest_loader, existing[idx])
                artifact_dir = make_artifact_dir(out_dir, run_name=preserved_name or run_name)
            except Exception as exc:
                print(f"Could not parse rewrite choice: {exc}; creating a new run")
                artifact_dir = None

    if artifact_dir is None:
        artifact_dir = make_artifact_dir(out_dir, run_name=run_name)

    return TrainingRunContext(artifact_dir=artifact_dir, resume_checkpoint=resume_ckpt)
