# features/embedder.py
import numpy as np
from sklearn.decomposition import PCA

class Embedder:
    def pca(self, X: np.ndarray, n=2):
        p = PCA(n_components=n, random_state=0)
        Z = p.fit_transform(X)
        return Z, p
