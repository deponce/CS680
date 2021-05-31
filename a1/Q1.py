from model.models import *
from matplotlib import pyplot as plt

def main():
    X_train_data = np.genfromtxt('./data/spambase_X.csv', delimiter=",")
    Y_train_data = np.genfromtxt('./data/spambase_y.csv')
    max_pass = 500
    model = models.perceptron(max_pass=max_pass)
    model.fit(X_train_data, Y_train_data)
    weight, bias, mistakes = model._weight, model._bias, model._loss
    plt.title("mistakes w.r.t passes")
    plt.xlabel("passes")
    plt.ylabel("# mistakes")
    x_data = [i+1 for i in range(max_pass)]
    plt.plot(x_data, mistakes)
    plt.grid(1)
    plt.show()

if __name__ == '__main__':
    main()
