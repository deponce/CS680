import numpy as np
from matplotlib import pyplot as plt

def svrGD(X, y, max_pass, eta):
    w = np.zeros(len(X[0]))
    b = 0
    train_loss = []
    train_error = []
    for t in range(max_pass):
        for i in range(len(X)):
            if abs(y[i] - w.T.dot(X[i]) - b) >= 0.5:
                if y[i] - w.T.dot(X[i]) - b > 0.5:
                    w = w + eta * X[i]
                    b = b + eta * 1
                else:
                    w = w - eta * X[i]
                    b = b - eta * 1
            w = w / (1 + eta)
        train_loss.append(compute_loss(X, y, w, b, 1, 0.5))
        train_error.append(compute_error(X, y, w, b, 1, 0.5))
    return w, b, train_loss, train_error

def compute_loss(X, Y, w, b, C, eps):
    # Implement me!
    # Return: loss computed on the given set
    loss = 0
    for i in range(len(X)):
        # print(Y - w.T.dot(X[i]-b))
        loss += C * max(abs(Y[i] - w.T.dot(X[i]) - b) - eps, 0)
    return loss +  1/2 * np.linalg.norm(w)**2

def compute_error(X, Y, w, b, C, eps):
    #Implement me!
    #Return: error computed on the given set
    error = 0
    for i in range(len(X)):
        error += C * max(abs(Y[i] - w.T.dot(X[i]) - b) - eps, 0)
    return error

# print(np.array([[1,2],[3,4]]).dot(np.array([[1,2],[3,4]])))
X_train_C = np.loadtxt('./data/X_train_C.csv', delimiter=",")
Y_train_C = np.loadtxt('./data/Y_train_C.csv', delimiter=",").astype(int)
X_test_C = np.loadtxt('./data/X_test_C.csv', delimiter=",")
Y_test_C = np.loadtxt('./data/Y_test_C.csv', delimiter=",").astype(int)



w, b, train_loss, train_error = svrGD(X_train_C, Y_train_C, 500, 1e-4)
print('w: ', w)
print('b: ', b)
print(train_error[-1])
print(compute_error(X_test_C, Y_test_C, w, b, 1, 0.5))
print(len(train_loss))
plt.title("loss  VS.  num_of_passes")
plt.xlabel("num_of_passes")
plt.ylabel("loss")
plt.plot(np.arange(1, 500+1, 1), train_loss)
plt.show()