import numpy as np
from model.metrics import *
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

def ridge_regression_GD(X_train_data, Y_train_data, Lambda, Max_pass, learning_rate):
    X_train_shape = X_train_data.shape
    n_features = X_train_shape[0]
    n_train_data = X_train_shape[1]
    X_train_data_p = np.r_[X_train_data, np.ones((1, n_train_data))]
    X_train_data_p = X_train_data_p.T
    w = np.zeros(n_features+1)
    factor = 1
    losses = []
    E = np.eye(n_features+1)
    E[-1,-1]=0
    for i in range(Max_pass):

        w = w - learning_rate*factor*(
                (1/n_train_data)*(X_train_data_p.T.dot(X_train_data_p).dot(w) -
                X_train_data_p.T.dot(Y_train_data))+
                2*Lambda*E.dot(w)
        )
        losses.append(MSE_l1(w[:-1], w[-1], X_train_data, Y_train_data, alpha=1))
    return w[:-1], w[-1], losses

def ridge_regression_Newton_method(X_train_data, Y_train_data, Lambda, Max_pass, learning_rate):
    X_train_shape = X_train_data.shape
    n_features = X_train_shape[0]
    n_train_data = X_train_shape[1]
    X_train_data_p = np.r_[X_train_data, np.ones((1, n_train_data))]
    X_train_data_p = X_train_data_p.T
    w = np.zeros(n_features+1)
    factor = 1
    losses = []
    E = np.eye(n_features+1)
    E[-1,-1]=0
    for i in range(Max_pass):

        w = w - learning_rate*factor/n_train_data*(
                np.dot(np.linalg.inv(X_train_data_p.T.dot(X_train_data_p)+Lambda*E),
                (X_train_data_p.T.dot(X_train_data_p).dot(w)-
                X_train_data_p.T.dot(Y_train_data)+
                Lambda*E.dot(w)))
        )
        losses.append(MSE_l1(w[:-1], w[-1], X_train_data, Y_train_data, alpha=1))
    return w[:-1], w[-1], losses

