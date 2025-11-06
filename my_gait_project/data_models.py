from dataclasses import dataclass
import numpy as np
from typing import Dict, List

@dataclass
class Frame:
    t: float
    kp2d: Dict[str, np.ndarray]
    kp3d: Dict[str, np.ndarray]
    score: Dict[str, float]

@dataclass
class Trial:
    frames: List[Frame]

@dataclass
class Step:
    side: str
    start_i: int
    end_i: int
    resampled: Dict[str, np.ndarray]
