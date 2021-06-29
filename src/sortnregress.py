import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC


def sortnregress(X):
    """ Take n x d data, order nodes by marginal variance and
    regresses each node onto those with lower variance, using
    edge coefficients as structure estimates. """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion='bic')

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        LR.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, covariates] * weight, X[:, target].ravel())
        W[covariates, target] = LL.coef_ * weight

    return W


if __name__ == "__main__":
    W = np.array([[0, 1, 0], [0, 0, 2], [0, 0, 0]])
    X = np.random.randn(1000, 3).dot(np.linalg.inv(np.eye(3) - W))
    W_hat = sortnregress(X)
    print(W)
    print(W_hat)
