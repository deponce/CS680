import numpy as np
from src.models import *
from src.method import *
import time
from matplotlib import pyplot as plt
def main():
    def loss(Y,Y_hat):
        return np.linalg.norm((Y-Y_hat),2)**2
    X_train_D = np.genfromtxt('./data/X_train_D.csv', delimiter=",")
    Y_train_D = np.genfromtxt('./data/Y_train_D.csv', delimiter=",")

    X_test_D = np.genfromtxt('./data/X_test_D.csv', delimiter=",")
    Y_test_D = np.genfromtxt('./data/Y_test_D.csv', delimiter=",")

    X_train_E = np.genfromtxt('./data/X_train_E.csv', delimiter=",")
    Y_train_E = np.genfromtxt('./data/Y_train_E.csv', delimiter=",")

    X_test_E = np.genfromtxt('./data/X_test_E.csv', delimiter=",")
    Y_test_E = np.genfromtxt('./data/Y_test_E.csv', delimiter=",")
    model = models.KNN()
    model.fit(X_train_D, Y_train_D)
    for i in range(1,9):
        model.set_k(i)
        y_hat = model.predict(X_test_D)
        print(loss(y_hat,Y_test_D))


    """X_test_data = np.genfromtxt('./data/housing_X_test.csv', delimiter=",")
    X_train_data = np.genfromtxt('./data/housing_X_train.csv', delimiter=",")
    Y_test_data = np.genfromtxt('./data/housing_y_test.csv')
    Y_train_data = np.genfromtxt('./data/housing_y_train.csv')
    Lambda = 10
    Max_pass=100000
    learning_rate=7.5e-2
    model = models.ridge_regression_Newton_method(Lambda=Lambda,
                                       Max_pass=Max_pass,
                                       learning_rate=learning_rate)
    model.fit(X_train_data, Y_train_data)
    weight, bias, loss = model._weight, model._bias, model._loss
    print(weight, bias)
    plt.plot(loss)
    plt.show()
    """


if __name__ == '__main__':
    main()
