from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EpochMetrics:
    epoch: int
    loss: float
    rec_loss: float
    kld: float


@dataclass
class TrainingSummary:
    epochs: int
    metrics: List[EpochMetrics]
    final_model_path: str
    checkpoint_paths: List[str]


@dataclass
class Checkpoint:
    epoch: int
    model_state: Dict
    optimizer_state: Dict
    extras: Optional[Dict] = None
