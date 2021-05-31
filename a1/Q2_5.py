from model.metrics import *
from model.models import *
from matplotlib import pyplot as plt
import time
def output_loss(weight, bias, X_train_data, Y_train_data, X_test_data, Y_test_data):
    print("training error: ",MSE(weight, bias, X_train_data, Y_train_data))
    print("training loss: ",MSE_l1(weight, bias, X_train_data, Y_train_data, alpha=1))
    print("test error: ",MSE(weight, bias, X_test_data, Y_test_data))

def main():
    X_test_data = np.genfromtxt('./data/housing_X_test.csv', delimiter=",")
    X_train_data = np.genfromtxt('./data/housing_X_train.csv', delimiter=",")
    Y_test_data = np.genfromtxt('./data/housing_y_test.csv')
    Y_train_data = np.genfromtxt('./data/housing_y_train.csv')
    print("---------------------closed_form, lambda=0---------------------")
    model = models.ridge_regression_closed_form(Lambda=0)
    start = time.time()
    model.fit(X_train_data, Y_train_data)
    end = time.time()
    print("running time: ", end-start, "sec")
    output_loss(model._weight, model._bias, X_train_data, Y_train_data, X_test_data, Y_test_data)
    print("\n---------------------closed_form, lambda=10--------------------")
    model.set_lambda(Lambda=10)
    start = time.time()
    model.fit(X_train_data, Y_train_data)
    end = time.time()
    print("running time: ", end-start, "sec")
    output_loss(model._weight, model._bias, X_train_data, Y_train_data, X_test_data, Y_test_data)
    print("\n-------------------gradient descent, lambda=0------------------")
    max_pass = 100
    model = models.ridge_regression_GD(Lambda=0, Max_pass=max_pass,)
    start = time.time()
    model.fit(X_train_data, Y_train_data)
    end = time.time()
    print("running time: ", end - start, "sec")
    output_loss(model._weight, model._bias, X_train_data, Y_train_data, X_test_data, Y_test_data)
    plt.subplot(1, 2, 1)
    x_data = [i + 1 for i in range(max_pass)]
    plt.plot(x_data, model._loss)
    plt.grid(1)
    plt.title("MSE w.r.t passes, Lambda=0")
    plt.xlabel("passes")
    plt.ylabel("MSE")
    print("\n-------------------gradient descent, lambda=10-----------------")
    max_pass = 100
    model = models.ridge_regression_GD(Lambda=10, Max_pass=max_pass,)
    start = time.time()
    model.fit(X_train_data, Y_train_data)
    end = time.time()
    print("running time: ", end - start, "sec")
    output_loss(model._weight, model._bias, X_train_data, Y_train_data, X_test_data, Y_test_data)
    plt.subplot(1,2,2)
    x_data = [i + 1 for i in range(max_pass)]
    plt.plot(x_data, model._loss)
    plt.grid(1)
    plt.title("MSE w.r.t passes, Lambda=10")
    plt.xlabel("passes")
    plt.ylabel("MSE")
    plt.show()
if __name__ == '__main__':
    main()
