import numpy as np

def varsortability(X, W, tol=1e-9):
    """ Takes n x d data and a d x d adjaceny matrix,
    where the i,j-th entry corresponds to the edge weight for i->j,
    and returns a value indicating how well the variance order
    reflects the causal order. """
    E = W != 0
    Ek = E.copy()
    var = np.var(X, axis=0, keepdims=True)

    n_paths = 0
    n_correctly_ordered_paths = 0

    for _ in range(E.shape[0] - 1):
        n_paths += Ek.sum()
        n_correctly_ordered_paths += (Ek * var / var.T > 1 + tol).sum()
        n_correctly_ordered_paths += 1/2*(
            (Ek * var / var.T <= 1 + tol) *
            (Ek * var / var.T >  1 - tol)).sum()
        Ek = Ek.dot(E)

    return n_correctly_ordered_paths / n_paths

if __name__ == "__main__":
    W = np.array([[0, 1, 0], [0, 0, 2], [0, 0, 0]])
    X = np.random.randn(1000, 3).dot(np.linalg.inv(np.eye(3) - W))
    print("Varsortability:", varsortability(X, W))
    
    X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    print("Varsortability standardized:", varsortability(X_std, W))