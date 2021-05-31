import numpy as np

def MSE(w, x, y):
    n_data = y.shape
    return np.linalg.norm((x.dot(w)) - y, ord=2) ** 2/(2*n_data)

def MSE_l1(w, x, y, alpha=1):
    n_data = y.shape
    E_shape = w.shape
    E = np.eye(E_shape[0])
    E[-1, -1] = 0
    Ew = E.dot(w)
    return np.linalg.norm((x.dot(w)) - y, ord=2) ** 2/(2*n_data) + alpha*Ew.T.dot(Ew)
