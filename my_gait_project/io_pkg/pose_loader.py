# io_pkg/pose_loader.py
import json
import numpy as np
from typing import Dict, Any, List
from my_gait_project.data_models import Frame, Trial

class PoseLoader:
    def load_json(self, path: str) -> Trial:
        raw = self._load_list_of_frames(path)
        frames = [self._to_frame(item) for item in raw]
        frames = [f for f in frames if f is not None]
        frames.sort(key=lambda f: f.t)
        return Trial(frames=frames)

    def _load_list_of_frames(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _to_frame(self, item: Dict[str, Any]) -> Frame | None:
        if not isinstance(item, dict) or len(item) != 1:
            return None
        t_str, payload = list(item.items())[0]
        try:
            t = float(t_str)
        except Exception:
            return None

        kp2d = self._keypoints_to_map(payload.get("keypoints2D", []), dims=2)
        kp3d = self._keypoints_to_map(payload.get("keypoints3D", []), dims=3)
        score = self._scores_map(payload)
        return Frame(t=t, kp2d=kp2d, kp3d=kp3d, score=score)

    def _keypoints_to_map(self, lst: List[Dict[str, Any]], dims: int) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for kp in lst:
            name = kp.get("name")
            if not name:
                continue
            if dims == 2:
                vec = np.array([kp.get("x"), kp.get("y")], dtype=float)
            else:
                vec = np.array([kp.get("x"), kp.get("y"), kp.get("z")], dtype=float)
            out[name] = vec
        return out

    def _scores_map(self, payload: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for part in ("keypoints2D", "keypoints3D"):
            for kp in payload.get(part, []):
                name = kp.get("name")
                sc = kp.get("score")
                if name is not None and sc is not None:
                    out[name] = float(sc)
        return out
