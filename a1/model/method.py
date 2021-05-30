import numpy as np

def perceptron(X_data,Y_data,max_pass):
    n_train_features = X_data.shape[0]
    n_train_data = X_data.shape[1]
    X_data = np.r_[X_data,np.ones((1,n_train_data))]
    X_data = X_data.T
    w = np.zeros(n_train_features+1)
    mistakes = [0 for _ in range(max_pass)]
    for t in range(max_pass):
        mistake = 0
        for idx, x in enumerate(X_data):
            if Y_data[idx]*(np.dot(x, w)) <= 0:
                w += Y_data[idx]*x
                mistake += 1
        mistakes[t]=mistake
    return w[:-1], w[-1], mistakes

def ridge_regression_closed_form(X_train_data, Y_train_data, Lambda=1):
    #X_test_shape = X_test_data.shape
    X_train_shape = X_train_data.shape
    n_features = X_train_data.shape[0]
    n_train_data = X_train_shape[1]
    X_train_data = np.r_[X_train_data, np.ones((1, n_train_data))]
    eyes = 2*n_train_data*Lambda * np.eye(n_features + 1)
    eyes[-1, -1] = 0
    ATA = np.matmul(X_train_data, X_train_data.T) + eyes
    ATZ = np.matmul(X_train_data, Y_train_data)
    W = np.linalg.solve(ATA, ATZ)
    return W[:-1], W[-1]