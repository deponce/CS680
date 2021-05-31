import numpy as np
import model.method as method
import time
class models:
    def __init__(self):
        self.perceptron = self.perceptron()
        self.ridge_regression_closed_form = self.ridge_regression_closed_form()
        self.ridge_regression_GD = self.ridge_regression_GD()
        self.ridge_regression_Newton_method = self.ridge_regression_Newton_method()
        self.KNN = self.KNN()

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

        def __init__(self, Lambda=0):
            self.Lambda = Lambda

        def set_lambda(self, Lambda=0):
            self.Lambda = Lambda

        def fit(self, X, Y):
            self._weight, self._bias = method.ridge_regression_closed_form(X, Y, self.Lambda)

    class ridge_regression_GD:
        def __init__(self, Lambda=0, Max_pass=500, learning_rate=1e-6):
            self.Lambda = Lambda
            self.Max_pass = Max_pass
            self.learning_rate = learning_rate
            self._loss = np.array([])

        def set_lambda(self, Lambda=0):
            self.Lambda = Lambda

        def fit(self, X, Y):
            self._weight, self._bias, self._loss = method.ridge_regression_GD(X, Y, self.Lambda, self.Max_pass, self.learning_rate)

    class ridge_regression_Newton_method:
        def __init__(self, Lambda=0, Max_pass=100000, learning_rate=7.5e-2):
            self.Lambda = Lambda
            self.Max_pass = Max_pass
            self.learning_rate = learning_rate
            self._loss = np.array([])

        def set_lambda(self, Lambda=0):
            self.Lambda = Lambda

        def fit(self,X, Y):
            self._weight, self._bias, self._loss = method.ridge_regression_Newton_method(X, Y, self.Lambda, self.Max_pass, self.learning_rate)

        def predict(self,test_X):
            pass

    class KNN:
        def __init__(self, k=1):
            self.k = k
            self.train_X = np.array([])
            self.train_Y = np.array([])

        def fit(self,X, Y):
            self.train_X = X
            self.train_Y = Y

        def set_k(self, k):
            self.k = k

        def predict(self, test_X):
            def l2_distance(x1, x2):
                return (x1 - x2) ** 2
            Y_hat = []
            dis_array = np.array([float('inf') for _ in range(len(self.train_X))])
            for  test_x in test_X:
                for train_idx, train_x in enumerate(self.train_X):
                    dis_array[train_idx] = l2_distance(train_x,test_x)
                indices = dis_array.argsort()[:self.k]
                Y_hat.append(np.mean(self.train_Y[indices]))
            return np.array(Y_hat)
