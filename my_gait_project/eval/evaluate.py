# eval/evaluate.py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

def quick_auc_acc(X: np.ndarray, y: np.ndarray, n_splits=5):
    aucs, accs = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], (p>0.5).astype(int)))
    return float(np.mean(aucs)), float(np.mean(accs))
