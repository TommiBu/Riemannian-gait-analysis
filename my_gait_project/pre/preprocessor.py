# pre/preprocessor.py
import numpy as np
from my_gait_project.data_models import Trial

class Preprocessor:
    def __init__(self, use_3d: bool = False):
        self.use_3d = use_3d

    def center_on_pelvis(self, trial: Trial) -> Trial:
        for fr in trial.frames:
            L = fr.kp3d if self.use_3d else fr.kp2d
            hipL = L.get("left_hip"); hipR = L.get("right_hip")
            if hipL is None or hipR is None: continue
            pelvis = (hipL + hipR) / 2.0
            for name, v in list(L.items()):
                L[name] = v - pelvis
        return trial

    def compute_leg_length(self, trial: Trial, side: str = "left") -> float:
        vals = []
        for fr in trial.frames:
            L = fr.kp3d if self.use_3d else fr.kp2d
            hip = L.get(f"{side}_hip"); ankle = L.get(f"{side}_ankle")
            if hip is None or ankle is None: continue
            vals.append(np.linalg.norm(hip - ankle))
        return float(np.nanmedian(vals)) if vals else np.nan

    def scale_by_leg(self, trial: Trial, leg_len: float) -> Trial:
        if not np.isfinite(leg_len) or leg_len <= 0: return trial
        for fr in trial.frames:
            L = fr.kp3d if self.use_3d else fr.kp2d
            for name, v in list(L.items()):
                L[name] = v / leg_len
        return trial

    def smooth(self, trial: Trial, win: int = 5) -> Trial:
        # jednoduchý klouzavý průměr nad všemi klouby a dimenzemi
        half = max(1, win//2)
        arr = trial.frames
        for i in range(len(arr)):
            rng = arr[max(0,i-half):min(len(arr), i+half+1)]
            for name in arr[i].kp2d.keys():
                v = np.stack([f.kp2d[name] for f in rng if name in f.kp2d], axis=0)
                arr[i].kp2d[name] = np.nanmean(v, axis=0)
            if arr[i].kp3d:
                for name in arr[i].kp3d.keys():
                    v = np.stack([f.kp3d[name] for f in rng if name in f.kp3d], axis=0)
                    arr[i].kp3d[name] = np.nanmean(v, axis=0)
        return trial
