import numpy as np
import copy
from matplotlib import pyplot as plt

def SVR(X_train, Y_train, C, eps):
    #Implement me! You may choose other parameters eta, max_pass, etc. internally
    #Return: parameter vector w, b
    w = np.zeros(X_train.shape[1])
    b = np.zeros(1)
    max_pass = 500
    lr = 1e-4
    loss = []
    error = []
    last_loss = float("inf")
    for _ in range(max_pass):
        for i in range(Y_train.shape[0]):
            if np.abs(Y_train[i]-(X_train[i].dot(w)+b)) >= eps:
                cpw = copy.deepcopy(w)
                w = cpw - lr*(-np.sign(Y_train[i]-(X_train[i].dot(cpw)+b)-eps))*X_train[i]
                b = b - lr*C*(-np.sign(Y_train[i]-(X_train[i].dot(cpw)+b)-eps))
            w = w/(1+lr)

        current_loss = compute_loss(X_train, Y_train,w,b,C,eps)
        """   
        if abs(last_loss - current_loss)<0.01:
            lr = lr/1.414
            print(lr)
            """
        loss.append(compute_loss(X_train, Y_train,w,b,C,eps))
        last_loss = current_loss
        error.append(compute_error(X_train, Y_train,w,b,C,eps))
    return loss, error

def compute_loss(X, Y, w, b, C, eps):
    err = compute_error(X, Y, w, b, C, eps)
    loss = err + 1/2*w.T.dot(w)
    return loss

def compute_error(X, Y, w, b, C, eps):
    err = C*np.sum(np.maximum(np.abs(Y-X.dot(w)-b)-eps,0))
    return err
X_test_C = np.genfromtxt('./data/X_test_C.csv', delimiter=",")
Y_test_C = np.genfromtxt('./data/Y_test_C.csv', delimiter=",")

X_train_C = np.genfromtxt('./data/X_train_C.csv', delimiter=",")
Y_train_C = np.genfromtxt('./data/Y_train_C.csv', delimiter=",")

C = 1
eps = 0.5

#print(SVR(X_train_C, Y_train_C, C, eps))
loss, error = SVR(X_train_C, Y_train_C, C, eps)
plt.plot(error)
plt.plot(loss)
plt.show()
