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

        def predict(self,test_X):
            return test_X.T.dot(self._weight) + self._bias

    class ridge_regression_closed_form:
        _weight = np.array([])
        _bias = np.array([])

        def __init__(self, Lambda=0):
            self.Lambda = Lambda

        def set_lambda(self, Lambda=0):
            self.Lambda = Lambda

        def fit(self, X, Y):
            self._weight, self._bias = method.ridge_regression_closed_form(X, Y, self.Lambda)

        def predict(self, test_X):
            return test_X.T.dot(self._weight) + self._bias

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

        def predict(self, test_X):
            return test_X.T.dot(self._weight) + self._bias + self._bias

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
            return test_X.T.dot(self._weight) + self._bias

    class KNN:
        def __init__(self, k=1):
            self.k = k
            self.train_X = np.array([])
            self.train_Y = np.array([])
            self.train_pairs = []
        class data_pair:
            def __init__(self, X, Y, dist):
                self.x = X
                self.y = Y
                self.dist = dist
        def k_mean(self, arra, k):
            cnt = 0
            for i in range(k):
                cnt += arra[i].y
            return cnt/k

        def fit(self,X, Y):
            self.train_X = X.T
            self.train_Y = Y
            n_train = len(Y)
            train = []
            for i in range(n_train):
                train.append(self.data_pair(self.train_X[i],self.train_Y[i], 0))
            self.train_pairs = train

        def kthSmallest(self,arr, l, r, k):
            def partition(arr, l, r):
                x = arr[r].dist
                i = l
                for j in range(l, r):
                    if arr[j].dist <= x:
                        arr[i], arr[j] = arr[j], arr[i]
                        i += 1
                arr[i], arr[r] = arr[r], arr[i]
                return i
            if (k > 0 and k <= r - l + 1):

                index = partition(arr, l, r)

                if (index - l == k - 1):
                    return arr[index]
                if (index - l > k - 1):
                    return self.kthSmallest(arr, l, index - 1, k)
                return self.kthSmallest(arr, index + 1, r,
                                   k - index + l - 1)

        def set_k(self, k):
            self.k = k

        def predict(self, test_X):
            def l2_distance(x1, x2):
                return np.sum((x1-x2)**2)
            Y_hat = []
            n_train = len(self.train_pairs)
            for test_x in test_X.T:
                for train_idx in range(n_train):
                    self.train_pairs[train_idx].dist = l2_distance(test_x, self.train_pairs[train_idx].x)
                self.kthSmallest(self.train_pairs, 0, n_train-1, self.k)
                Y_hat.append(self.k_mean(self.train_pairs, self.k))
            return np.array(Y_hat)
