# gait/step_detector.py
import numpy as np
from scipy.signal import find_peaks
from my_gait_project.data_models import Trial, Step

class StepDetector:
    def __init__(self, fs: float, side: str = "left", min_step_s: float = 0.45):
        self.fs = fs
        self.side = side
        self.min_step_s = min_step_s

    def _signal_y(self, trial: Trial, joint: str) -> np.ndarray:
        return np.array([fr.kp2d[joint][1] for fr in trial.frames if joint in fr.kp2d], float)

    def detect_events(self, trial: Trial, joint: str = None):
        joint = joint or f"{self.side}_ankle"
        y = self._signal_y(trial, joint)
        inv = -y
        distance = max(1, int(self.min_step_s * self.fs))
        idx, _ = find_peaks(inv, distance=distance)
        return idx

    def cut_steps(self, events: np.ndarray):
        steps = []
        for a, b in zip(events[:-1], events[1:]):
            if b - a > 3:
                steps.append((int(a), int(b)))
        return steps

    def resample_steps(self, trial: Trial, spans, joints, n_points=101):
        out = []
        for (s, e) in spans:
            resampled = {}
            for j in joints:
                if j not in trial.frames[s].kp2d: continue
                xy = np.array([fr.kp2d[j] for fr in trial.frames[s:e]], float)  # (T,2)
                t_old = np.linspace(0, 1, len(xy))
                t_new = np.linspace(0, 1, n_points)
                x = np.interp(t_new, t_old, xy[:,0])
                y = np.interp(t_new, t_old, xy[:,1])
                resampled[j] = np.stack([x, y], axis=1)
            out.append(Step(side=self.side, start_i=s, end_i=e, resampled=resampled))
        return out
