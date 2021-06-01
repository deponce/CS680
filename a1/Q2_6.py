from model.metrics import *
from model.models import *
from sklearn.linear_model import Ridge

def ridge_regression_closedform(X_train_data, Y_train_data, Lambda=1):
    X_train_shape = X_train_data.shape
    n_features = X_train_data.shape[0]
    n_train_data = X_train_shape[1]
    X_train_data = np.r_[X_train_data, np.ones((1, n_train_data))]
    eyes = Lambda * np.eye(n_features + 1)
    eyes[-1, -1] = 0
    ATA = np.matmul(X_train_data, X_train_data.T) + eyes
    ATZ = np.matmul(X_train_data, Y_train_data)
    W = np.linalg.solve(ATA, ATZ)
    return W[:-1], W[-1]

def main():
    print("###################################Q2_6#########################################")
    X_train_data = np.genfromtxt('./data/housing_X_train.csv', delimiter=",")
    Y_train_data = np.genfromtxt('./data/housing_y_train.csv')
    print("-------------------Newton's method, lambda=0------------------")
    max_pass = 50000
    model = models.ridge_regression_Newton_method(Lambda=0, Max_pass=max_pass,)
    model.fit(X_train_data, Y_train_data)
    print("weight: \n", model._weight)
    print("bias: \n", model._bias)
    print("---------------------closed_form, lambda=0---------------------")
    weight, bias = ridge_regression_closedform(X_train_data, Y_train_data, Lambda=0)
    print("weight: \n", weight)
    print("bias: \n", bias)
    print("\n-----------------------Sklearn, lambda=0----------------------")
    reg = Ridge(alpha=0).fit(X_train_data.T,Y_train_data)
    print("weight: \n", reg.coef_)
    print("bias: \n", reg.intercept_)
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------\n")
    print("\n-------------------Newton's method, lambda=10-----------------")
    model = models.ridge_regression_Newton_method(Lambda=10, Max_pass=max_pass,)
    model.fit(X_train_data, Y_train_data)
    print("weight: \n", model._weight)
    print("bias: \n", model._bias)
    print("\n---------------------closed_form, lambda=10--------------------")
    weight, bias = ridge_regression_closedform(X_train_data, Y_train_data, Lambda=10)
    print("weight: \n", weight)
    print("bias: \n", bias)
    print("\n-----------------------Sklearn, lambda=10---------------------")
    reg = Ridge(alpha=10).fit(X_train_data.T, Y_train_data)
    print("weight: \n", reg.coef_)
    print("bias: \n", reg.intercept_)
    print("###################################Q2_6#########################################")
if __name__ == '__main__':
    main()
