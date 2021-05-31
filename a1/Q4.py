from model.models import *
from model.metrics import *
from matplotlib import pyplot as plt

def output_loss(weight, bias, X_train_data, Y_train_data, X_test_data, Y_test_data):
    print("training error: ",MSE(weight, bias, X_train_data, Y_train_data))
    print("training loss: ",MSE_l1(weight, bias, X_train_data, Y_train_data, alpha=1))
    print("test error: ",MSE(weight, bias, X_test_data, Y_test_data))

def main():
    X_train_D = np.genfromtxt('./data/X_train_D.csv', delimiter=",").reshape((1,-1))

    Y_train_D = np.genfromtxt('./data/Y_train_D.csv', delimiter=",")

    X_test_D = np.genfromtxt('./data/X_test_D.csv', delimiter=",").reshape((1,-1))
    Y_test_D = np.genfromtxt('./data/Y_test_D.csv', delimiter=",")

    X_train_E = np.genfromtxt('./data/X_train_E.csv', delimiter=",")
    Y_train_E = np.genfromtxt('./data/Y_train_E.csv', delimiter=",")

    X_test_E = np.genfromtxt('./data/X_test_E.csv', delimiter=",")
    Y_test_E = np.genfromtxt('./data/Y_test_E.csv', delimiter=",")

    print("-----------------------------dataset E-----------------------------")
    model = models.ridge_regression_closed_form(Lambda=0)
    model.fit(X_train_D, Y_train_D)
    #print(model.predict(X_test_D))

    model = models.KNN()
    model.fit(X_train_D, Y_train_D)
    for k in range(1,10):
        model.set_k(k)
        model.fit(X_train_D, Y_train_D)
        y_hat=model.predict(X_test_D)
        print(k,y_hat)




if __name__ == '__main__':
    main()
