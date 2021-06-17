import sys
import numpy as np
import copy
#Exercise 3
#Usage: python3 ex3.py X_train Y_train X_test Y_test C eps

def SVR(X_train, Y_train, C, eps):
    #Implement me! You may choose other parameters eta, max_pass, etc. internally
    #Return: parameter vector w, b
    w = np.zeros(X_train.shape[1])
    b = np.zeros(1)
    max_pass = 500
    lr = 1e-3
    for _ in range(max_pass):
        for i in range(Y_train.shape[0]):
            if np.abs(Y_train[i]-(X_train[i].dot(w)+b)) >= eps:
                cpw = copy.deepcopy(w)
                w = cpw - lr*(-np.sign(Y_train[i]-(X_train[i].dot(cpw)+b))*X_train[i])
                b = b - lr*C*(-np.sign(Y_train[i]-(X_train[i].dot(cpw)+b)))
            w = w/(1+lr)
    return w, b

def compute_loss(X, Y, w, b, C, eps):
    err = compute_error(X, Y, w, b, C, eps)
    loss = err + 1/2*w.T.dot(w)
    return loss

def compute_error(X, Y, w, b, C, eps):
    err = C*np.sum(np.maximum(np.abs(Y-X.dot(w)-b)-eps,0))
    return err


if __name__ == "__main__":
    args = sys.argv[1:]
    #You may import the data some other way if you prefer

    X_train = np.loadtxt(args[0], delimiter=",")
    Y_train = np.loadtxt(args[1], delimiter=",")
    X_test = np.loadtxt(args[2], delimiter=",")
    Y_test = np.loadtxt(args[3], delimiter=",")
    C = float(args[4])
    eps = float(args[5])

    w, b = SVR(X_train, Y_train, C, eps)
    print("training loss: ",compute_loss(X_train, Y_train, w, b, C, eps))
    print("training error: ",compute_error(X_train, Y_train, w, b, C, eps))
    print("test error: ",compute_error(X_test, Y_test, w, b, C, eps))
