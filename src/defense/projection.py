import numpy as np


def random_basis(dim, rank, seed=42):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
    return q[:, :rank]


def pca_basis(updates, rank):
    x = np.asarray(updates, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[:rank].T
