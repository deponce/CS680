from model.models import *
from model.metrics import *
from matplotlib import pyplot as plt

def loss(Y,Y_hat):
    return np.linalg.norm((Y-Y_hat),2)**2/len(Y_hat)

def output_loss(weight, bias, X_train_data, Y_train_data, X_test_data, Y_test_data):
    print("training error: ",MSE(weight, bias, X_train_data, Y_train_data))
    print("training loss: ",MSE_l1(weight, bias, X_train_data, Y_train_data, alpha=1))
    print("test error: ",MSE(weight, bias, X_test_data, Y_test_data))

def main():
    X_train_D = np.genfromtxt('./data/X_train_D.csv', delimiter=",").reshape((1,-1))

    Y_train_D = np.genfromtxt('./data/Y_train_D.csv', delimiter=",")

    X_test_D = np.genfromtxt('./data/X_test_D.csv', delimiter=",").reshape((1,-1))
    Y_test_D = np.genfromtxt('./data/Y_test_D.csv', delimiter=",")

    X_train_E = np.genfromtxt('./data/X_train_E.csv', delimiter=",").reshape((1,-1))
    Y_train_E = np.genfromtxt('./data/Y_train_E.csv', delimiter=",")

    X_test_E = np.genfromtxt('./data/X_test_E.csv', delimiter=",").reshape((1,-1))
    Y_test_E = np.genfromtxt('./data/Y_test_E.csv', delimiter=",")

    print("-----------------------------dataset D---------------------------------")
    model = models.ridge_regression_closed_form(Lambda=0)
    model.fit(X_train_D, Y_train_D)
    y_hat = model.predict(X_test_D)
    mseloss = loss(Y_test_D,y_hat)
    plt.subplot(1,2,1)
    plt.title("dataset" + "D")
    plt.scatter(X_train_D,Y_train_D, c="darkslategray", alpha=0.8, label="Training set")
    plt.scatter(X_test_D, Y_test_D, c = "seagreen", alpha=0.5, label="test set")
    plt.plot(X_test_D[0], y_hat, c = "indigo", alpha=0.5, label="closed-form result")
    plt.xlabel("x value")
    plt.ylabel("y value")
    losses = []
    model = models.KNN()
    model.fit(X_train_D, Y_train_D)
    for k in range(1,10):
        model.set_k(k)
        model.fit(X_train_D, Y_train_D)
        y_hat=model.predict(X_test_D)
        losses.append(loss(Y_test_D,y_hat))
        if k in [1,9]:
            if k == 1:
                color = "coral"
                Lable = "1NN result"
            else:
                color = "firebrick"
                Lable = "9NN result"
            plt.scatter(X_test_D[0],y_hat, c = color, alpha=0.5, label=Lable)
    plt.legend()
    plt.subplot(1,2,2)
    plt.xlabel("k")
    plt.ylabel("mean-square error")
    plt.plot([i for i in range(1,10)],[mseloss for _ in range(1,10)])
    K = [i for i in range(1,10)]
    plt.plot(K, losses)
    plt.show()
    print("-----------------------------dataset E---------------------------------")
    model = models.ridge_regression_closed_form(Lambda=0)
    model.fit(X_train_E, Y_train_E)
    y_hat = model.predict(X_test_E)
    mseloss = loss(Y_test_E,y_hat)
    plt.subplot(1,2,1)
    plt.title("dataset" + "E")
    plt.scatter(X_train_E,Y_train_E, c="darkslategray", alpha=0.8, label="Training set")
    plt.scatter(X_test_E, Y_test_E, c = "seagreen", alpha=0.5, label="test set")
    plt.plot(X_test_E[0], y_hat, c = "indigo", alpha=0.5, label="closed-form result")
    plt.xlabel("x value")
    plt.ylabel("y value")
    losses = []
    model = models.KNN()
    model.fit(X_train_E, Y_train_E)
    for k in range(1,10):
        model.set_k(k)
        model.fit(X_train_E, Y_train_E)
        y_hat=model.predict(X_test_E)
        losses.append(loss(Y_test_E,y_hat))
        if k in [1,9]:
            if k == 1:
                color = "coral"
                Lable = "1NN result"
            else:
                color = "firebrick"
                Lable = "9NN result"
            plt.scatter(X_test_E[0],y_hat, c = color, alpha=0.5, label=Lable)
    plt.legend()
    plt.subplot(1,2,2)
    plt.xlabel("k")
    plt.ylabel("mean-square error")
    plt.plot([i for i in range(1,10)],[mseloss for _ in range(1,10)])
    K = [i for i in range(1,10)]
    plt.plot(K, losses)
    plt.show()



if __name__ == '__main__':
    main()
