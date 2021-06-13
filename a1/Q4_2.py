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
    print("###################################Q4_2#########################################")
    X_train_D = np.genfromtxt('./data/X_train_D.csv', delimiter=",").reshape((1,-1))

    Y_train_D = np.genfromtxt('./data/Y_train_D.csv', delimiter=",")

    X_test_D = np.genfromtxt('./data/X_test_D.csv', delimiter=",").reshape((1,-1))
    Y_test_D = np.genfromtxt('./data/Y_test_D.csv', delimiter=",")
    min_x_D = np.min(np.append(X_train_D,X_test_D))
    max_x_D = np.max(np.append(X_train_D, X_test_D))
    plot_p_n=1000
    plot_x_D = np.arange(min_x_D, max_x_D, (max_x_D-min_x_D)/plot_p_n).reshape(1,-1)

    X_train_E = np.genfromtxt('./data/X_train_E.csv', delimiter=",").reshape((1,-1))
    Y_train_E = np.genfromtxt('./data/Y_train_E.csv', delimiter=",")

    X_test_E = np.genfromtxt('./data/X_test_E.csv', delimiter=",").reshape((1,-1))
    Y_test_E = np.genfromtxt('./data/Y_test_E.csv', delimiter=",")
    min_x_E = np.min(np.append(X_train_E,X_test_E))
    max_x_E = np.max(np.append(X_train_E, X_test_E))
    plot_x_E = np.arange(min_x_E, max_x_E, (max_x_E-min_x_E)/plot_p_n).reshape(1,-1)


    print("-----------------------------dataset D---------------------------------")
    model = models.ridge_regression_closed_form(Lambda=0)
    model.fit(X_train_D, Y_train_D)
    y_hat = model.predict(X_test_D)
    mseloss = loss(Y_test_D,y_hat)
    y_hat = model.predict(plot_x_D)
    plt.subplot(1,2,1)
    plt.title("least square solution, 1NN solution, and 9NN soultion on dataset D")
    plt.scatter(X_train_D,Y_train_D, c="darkslategray", alpha=1, label="Training set")
    #plt.scatter(X_test_D, Y_test_D, c = "seagreen", alpha=0.5, label="test set",edgecolors="seagreen")
    plt.plot(plot_x_D[0], y_hat, c = "indigo", alpha=0.5, label="closed-form result")
    plt.xlabel("x value")
    plt.ylabel("y value")
    losses = []
    model = models.KNN()
    model.fit(X_train_D, Y_train_D)
    for k in range(1,10):
        model.set_k(k)
        y_hat=model.predict(X_test_D)
        losses.append(loss(Y_test_D,y_hat))
    #plot k = 1
    model.set_k(1)
    y_hat=model.predict(plot_x_D)
    color = "coral"
    Lable = "1NN result"
    plot_index = plot_x_D[0].argsort()
    plt.plot(plot_x_D[0][plot_index], y_hat[plot_index], c=color, alpha=1, label=Lable)
    #plot k = 9
    model.set_k(9)
    y_hat=model.predict(plot_x_D)
    color = "firebrick"
    Lable = "9NN result"
    plot_index = plot_x_D[0].argsort()
    plt.plot(plot_x_D[0][plot_index], y_hat[plot_index], c=color, alpha=1, label=Lable)
    plt.legend()
    plt.subplot(1,2,2)
    plt.xlabel("k")
    plt.ylabel("mean-square error")
    plt.title("least square error, and KNN error VS K on test dataset D")
    plt.plot([i for i in range(1,10)],[mseloss for _ in range(1,10)], label='MSE of least square linear regression',color="coral")
    K = [i for i in range(1,10)]
    plt.plot(K, losses, label='MSE of KNN', color="seagreen")
    plt.legend()
    plt.show()
    print("-----------------------------dataset E---------------------------------")
    model = models.ridge_regression_closed_form(Lambda=0)
    model.fit(X_train_E, Y_train_E)
    y_hat = model.predict(X_test_E)
    mseloss = loss(Y_test_E,y_hat)
    plt.subplot(1,2,1)
    plt.title("least square solution, 1NN solution, and 9NN soultion on dataset E")
    plt.scatter(X_train_E,Y_train_E, c="darkslategray", alpha=0.8, label="Training set")
    #plt.scatter(X_test_E, Y_test_E, c="seagreen", alpha=0.5, label="test set", edgecolors="seagreen")

    plt.plot(X_test_E[0], y_hat, c="indigo", alpha=1, label="closed-form result")

    plt.xlabel("x value")
    plt.ylabel("y value")
    losses = []
    model = models.KNN()
    model.fit(X_train_E, Y_train_E)
    for k in range(1,10):
        model.set_k(k)
        model.fit(X_train_E, Y_train_E)
        y_hat=model.predict(X_test_E)
        losses.append(loss(Y_test_E, y_hat))
    # plot k = 1
    model.set_k(1)
    y_hat = model.predict(plot_x_E)
    color = "coral"
    Lable = "1NN result"
    plot_index = plot_x_E[0].argsort()
    plt.plot(plot_x_E[0][plot_index], y_hat[plot_index], c=color, alpha=1, label=Lable)
    # plot k = 9
    model.set_k(9)
    y_hat = model.predict(plot_x_E)
    color = "firebrick"
    Lable = "9NN result"
    plot_index = plot_x_E[0].argsort()
    plt.plot(plot_x_E[0][plot_index], y_hat[plot_index], c=color, alpha=1, label=Lable)
    plt.legend()
    plt.subplot(1,2,2)
    plt.xlabel("k")
    plt.ylabel("mean-square error")
    plt.title("least square error, and KNN error VS K on test dataset E")
    plt.plot([i for i in range(1,10)],[mseloss for _ in range(1,10)], label='MSE of least square linear regression',color="coral")
    K = [i for i in range(1,10)]
    plt.plot(K, losses, label='MSE of KNN', color="seagreen")
    plt.legend()
    plt.show()
    print("###################################Q4_2#########################################")

if __name__ == '__main__':
    main()
