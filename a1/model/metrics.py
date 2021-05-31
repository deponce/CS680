import numpy as np

def MSE(W, b, x, y):
    n_train_data = x.shape[1]
    x = np.r_[x, np.ones((1, n_train_data))]
    w = np.append(W, b)
    n_data = y.shape[0]
    return np.linalg.norm((x.T.dot(w)) - y, ord=2) ** 2/(2 * n_data)


def MSE_l1(W, b, x, y, alpha=1):
    n_train_data = x.shape[1]
    x = np.r_[x, np.ones((1, n_train_data))]
    w = np.append(W, b)
    n_data = y.shape[0]
    E_shape = w.shape
    E = np.eye(E_shape[0])
    E[-1, -1] = 0
    Ew = E.dot(w)
    tmp = np.linalg.norm((x.T.dot(w)) - y, ord=2) ** 2
    return np.linalg.norm((x.T.dot(w)) - y, ord=2) ** 2/(2*n_data) + alpha*Ew.T.dot(Ew)
