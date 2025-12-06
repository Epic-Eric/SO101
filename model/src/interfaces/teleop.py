from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class TeleopSample:
    timestamp: float
    action: Dict[str, float]
    observation: Dict[str, Any]  # may include images (np.ndarray HxWx3) and joint states

@dataclass
class TeleopSessionResult:
    samples: List[TeleopSample]
    metadata: Dict[str, Any]
    # optional path where artifacts were saved
    output_dir: Optional[str] = None
