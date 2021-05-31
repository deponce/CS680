from model.models import *
from model.metrics import *
from matplotlib import pyplot as plt

def loss(Y,Y_hat):
    return np.linalg.norm((Y-Y_hat),2)**2/len(Y_hat)

def main():
    X_train_F = np.genfromtxt('./data/X_train_F.csv', delimiter=",").T

    Y_train_F = np.genfromtxt('./data/Y_train_F.csv', delimiter=",")

    X_test_F = np.genfromtxt('./data/X_test_F.csv', delimiter=",").T
    Y_test_F = np.genfromtxt('./data/Y_test_F.csv', delimiter=",")

    model = models.ridge_regression_closed_form(Lambda=0)
    model.fit(X_train_F, Y_train_F)
    y_hat = model.predict(X_test_F)
    mseloss = loss(Y_test_F,y_hat)
    losses = []
    model = models.KNN()
    model.fit(X_train_F, Y_train_F)

    for k in range(1,10):
        model.set_k(k)
        y_hat=model.predict(X_test_F)
        losses.append(loss(Y_test_F, y_hat))
    plt.xlabel("k")
    plt.ylabel("mean-square error")
    plt.title("least square error, and KNN error VS K on test dataset F")
    plt.plot([i for i in range(1,10)], [mseloss for _ in range(1,10)], label='MSE of KNN', color="coral")
    K = [i for i in range(1,10)]
    plt.plot(K, losses, label='MSE of least square linear regression', color="seagreen")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main()