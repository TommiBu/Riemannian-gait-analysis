# main.py
import numpy as np
from io_pkg.pose_loader import PoseLoader
from pre.preprocessor import Preprocessor
from gait.step_detector import StepDetector
from features.feature_maker import FeatureMaker
from features.embedder import Embedder
from features.metrics import Metrics

JOINTS = ["left_hip","left_knee","left_ankle","left_shoulder"]

def estimate_fs(trial):
    t = np.array([f.t for f in trial.frames], float)
    dt = np.diff(t)
    return float(1/np.nanmean(dt)) if dt.size else 60.0

def main():
    loader = PoseLoader()
    trial = loader.load_json("data/run_mid_data.json")

    pre = Preprocessor(use_3d=False)
    trial = pre.center_on_pelvis(trial)
    L = pre.compute_leg_length(trial, side="left")
    trial = pre.scale_by_leg(trial, L)
    trial = pre.smooth(trial, win=5)

    fs = estimate_fs(trial)
    det = StepDetector(fs=fs, side="left", min_step_s=0.45)
    events = det.detect_events(trial, joint="left_ankle")
    spans = det.cut_steps(events)
    steps = det.resample_steps(trial, spans, joints=JOINTS, n_points=101)

    fm = FeatureMaker()
    X = fm.fingerprint(steps, joints=JOINTS, axes=("x","y"))
    emb = Embedder()
    Z, pca = emb.pca(X, n=2)

    met = Metrics()
    # trajektorie po fázi v PCA: pro jednoduchost použijeme jen osu PC1 přes fázi:
    # volitelně: poskládej pro každý krok „trajektorii“ z PC1/PC2 přes fázi,
    # tady si ukážeme plynulost „po fázi“ z X v originálním prostoru:
    # (rychlá demonstrace – reálně si udělej PCA po fázové ose pro každý krok)
    print(f"Frames: {len(trial.frames)}, fs≈{fs:.1f} Hz, steps: {len(steps)}, leg≈{L:.3f}")

    # jednoduchý sanity print pro jeden krok: délka/zakřivení v PCA prostoru
    if len(steps) >= 1:
        # rekonstruuj trajektorii kroku v PCA: projekce segmentu X na PCA komponenty
        # (vezmeme vektor otisku kroku a přemapujeme na 2D; 101 bodů musíme složit zpět)
        # zjednodušeně: ukážeme rozptyl bodů Z (globální embedding) jako variabilitu
        var = met.variability_sd(Z)
        print(f"Variabilita kroků v PCA prostoru (SD eukl.): {var:.4f}")

if __name__ == "__main__":
    main()
