import os
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Best-effort YAML loader that returns {} if missing or unreadable."""
    if not os.path.isfile(path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        # PyYAML not installed; return empty and let CLI args drive
        return {}
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}
