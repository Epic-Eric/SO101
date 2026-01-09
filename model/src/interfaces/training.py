from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EpochMetrics:
    epoch: int
    loss: float
    rec_loss: float
    kld: float
    # optional validation loss for this epoch
    val_loss: Optional[float] = None
    # additional stabilization metrics
    one_step_mse: Optional[float] = None
    rollout_mse: Optional[float] = None
    latent_drift: Optional[float] = None
    kld_raw: Optional[float] = None
    beta: Optional[float] = None
    val_one_step_mse: Optional[float] = None
    rollout_horizon: Optional[int] = None
    gate_threshold: Optional[float] = None
    # action conditioning metrics
    contrastive_loss: Optional[float] = None
    action_sensitivity: Optional[float] = None
    latent_action_variance: Optional[float] = None


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
