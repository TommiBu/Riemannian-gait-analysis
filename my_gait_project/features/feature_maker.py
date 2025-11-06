# features/feature_maker.py
import numpy as np
from typing import List
from my_gait_project.data_models import Step

class FeatureMaker:
    def fingerprint(self, steps: List[Step], joints: list, axes=("x","y")) -> np.ndarray:
        ax_idx = {"x":0,"y":1}
        rows = []
        for s in steps:
            parts = []
            for j in joints:
                if j not in s.resampled: continue
                arr = s.resampled[j]                        # (101,2)
                take = [ax_idx[a] for a in axes]
                parts.append(arr[:, take].reshape(-1))
            if parts:
                rows.append(np.concatenate(parts))
        return np.vstack(rows) if rows else np.empty((0, len(joints)*101*len(axes)))
