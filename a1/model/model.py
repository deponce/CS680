import numpy as np
import method
from matplotlib import pyplot as plt
import time
class models:
    class perceptron:
        _weight = np.array([])
        _bias = np.array([])
        _loss = np.array([])
        def __init__(self, max_pass=500):
            self.max_pass = max_pass
        def fit(self, X, Y):
            self._weight, self._bias, self._loss = method.perceptron(X, Y, self.max_pass)

    class ridge_regression_closed_form:
        _weight = np.array([])
        _bias = np.array([])
        def __init__(self, Lambda):
            self.Lambda = Lambda
        def fit(self, X, Y, Lambda=0):
            _weight, _bias = method.ridge_regression_closed_form(X, Y, Lambda)




X_data = np.genfromtxt('../data/spambase_X.csv',delimiter = ",")
Y_data = np.genfromtxt('../data/spambase_y.csv')
max_pass = 500
Perceptron = models.perceptron(max_pass)
start = time.time()
Perceptron.fit(X_data, Y_data)
weight, bias, mistakes = Perceptron._weight, Perceptron._bias, Perceptron._loss
end = time.time()
plt.title("mistakes w.r.t passes")
plt.xlabel("passes")
plt.ylabel("# mistakes")
x_data = [i+1 for i in range(max_pass)]
plt.plot(x_data, mistakes)
plt.show()
print("running time:", end - start, "sec")
print("weight: ",weight)
print("bias: ",bias)