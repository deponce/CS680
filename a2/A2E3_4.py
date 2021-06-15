import numpy as np
def SVR(X_train, Y_train, C, eps):
    #Implement me! You may choose other parameters eta, max_pass, etc. internally
    #Return: parameter vector w, b
    pass

def compute_loss(X, Y, w, b, C, eps):
    #Implement me!
    #Return: loss computed on the given set
    pass

def compute_error(X, Y, w, b, C, eps):
    #Implement me!
    #Return: error computed on the given set
    pass
X_test_C = np.genfromtxt('./data/X_test_C.csv', delimiter=",")
Y_test_C = np.genfromtxt('./data/Y_test_C.csv', delimiter=",")

X_train_C = np.genfromtxt('./data/X_train_C.csv', delimiter=",")
Y_train_C = np.genfromtxt('./data/Y_train_C.csv', delimiter=",")

print(Y_train_C)

