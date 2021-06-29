import numpy as np


def varsortability(X, W):
    """ Takes n x d data and a d x d adjaceny matrix,
    where the i,j-th entry corresponds to the edge weight for i->j,
    and returns a value indicating how well the variance order
    reflects the causal order. """
    E = W != 0
    Ek = E.copy()
    var = np.var(X, axis=0, keepdims=True)
    tol = var.min() * 1e-9

    n_paths = 0
    n_correctly_ordered_paths = 0

    for k in range(E.shape[0] - 1):
        n_paths += Ek.sum()
        n_correctly_ordered_paths += (Ek * var / var.T > 1 + tol).sum()
        Ek = Ek.dot(E)

    return n_correctly_ordered_paths / n_paths


if __name__ == "__main__":
    W = np.array([[0, 1, 0], [0, 0, 2], [0, 0, 0]])
    X = np.random.randn(1000, 3).dot(np.linalg.inv(np.eye(3) - W))
    print("Varsortability:", varsortability(X, W))
