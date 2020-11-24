import numpy as np


def softmax(L: np.ndarray) -> np.ndarray:
    assert len(L.shape) == 2

    # Numerically stable softmax function
    m = np.amax(L, axis=1)[:, None]
    X = np.exp(L - m)
    P = X / X.sum(axis=1)[:, None]

    return P