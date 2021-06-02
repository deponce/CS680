import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
print("###################################Q3_1#########################################")
X_train_A = np.genfromtxt('./data/X_train_A.csv',delimiter = ",")
X_train_B = np.genfromtxt('./data/X_train_B.csv',delimiter = ",")
X_train_C = np.genfromtxt('./data/X_train_C.csv',delimiter = ",")
Train_data_X = [X_train_A,X_train_B,X_train_C]

Y_train_A = np.genfromtxt('./data/Y_train_A.csv',delimiter = ",")
Y_train_B = np.genfromtxt('./data/Y_train_B.csv',delimiter = ",")
Y_train_C = np.genfromtxt('./data/Y_train_C.csv',delimiter = ",")
Train_data_Y = [Y_train_A, Y_train_B, Y_train_C]

X_test_A = np.genfromtxt('./data/X_test_A.csv',delimiter = ",")
X_test_B = np.genfromtxt('./data/X_test_B.csv',delimiter = ",")
X_test_C = np.genfromtxt('./data/X_test_C.csv',delimiter = ",")
Test_data_X = [X_test_A, X_test_B, X_test_C]

Y_test_A = np.genfromtxt('./data/Y_test_A.csv',delimiter = ",")
Y_test_B = np.genfromtxt('./data/Y_test_B.csv',delimiter = ",")
Y_test_C = np.genfromtxt('./data/Y_test_C.csv',delimiter = ",")
Test_data_Y = [Y_test_A, Y_test_B, Y_test_C]

def cal_MSE(w,b,x,y):
    n = len(y)
    e = np.dot(x,w)+b-y
    return np.linalg.norm(e, ord = 2)**2/n

def get_Lasso_vector(X, Y,test_x,test_y, reg = 0.0):
    model = Lasso(alpha=reg)
    model.fit(X, Y)
    w = model.coef_
    b = model.intercept_
    error = mean_squared_error(test_y, model.predict(test_x))
    return np.r_[w, b], error

def get_Ridge_vector(X, Y,test_x,test_y, reg = 0.0):
    model = Ridge(alpha=reg)
    model.fit(X, Y)
    w = model.coef_
    b = model.intercept_
    error = mean_squared_error(test_y, model.predict(test_x))
    return np.r_[w, b], error

def get_LinearRegression_vector(X, Y,test_x,test_y):
    model = LinearRegression()
    model.fit(X, Y)
    w = model.coef_
    b = model.intercept_
    error = mean_squared_error(test_y, model.predict(test_x))
    return np.r_[w,b], error

dataset_name = ['A', 'B', 'C']
num_bins = 200
n_data = 300
for idx, X in enumerate(Train_data_X):
    print("dataset: ", dataset_name[idx])
    lrv,error = get_LinearRegression_vector(X, Train_data_Y[idx],Test_data_X[idx],Test_data_Y[idx])
    print("MES of LinearRegression: ", error)

    r1v,error = get_Ridge_vector(X, Train_data_Y[idx],Test_data_X[idx],Test_data_Y[idx], 1)
    print("MES of Ridge with reg 1: ", error)

    r10v,error = get_Ridge_vector(X, Train_data_Y[idx],Test_data_X[idx],Test_data_Y[idx], 10)
    print("MES of Ridge with reg 10: ", error)
    if dataset_name[idx]=='C':
        n_data = 50
    l1v,error = get_Lasso_vector(X, Train_data_Y[idx],Test_data_X[idx],Test_data_Y[idx], 1/n_data)
    print("MES of Lasso with reg 1: ", error)

    l10v,error = get_Lasso_vector(X, Train_data_Y[idx],Test_data_X[idx],Test_data_Y[idx], 10/n_data)
    print("MES of Lasso with reg 10: ", error)

    plot_data = (lrv[:-1], r1v[:-1], r10v[:-1], l1v[:-1], l10v[:-1])
    colors = ("indigo", "seagreen", "darkslategray", "coral", "firebrick")
    labels = ("LinearRegression", "Ridge with reg 1", "Ridge with reg 10", "Lasso with reg 1", "Lasso with reg 10")
    plt.hist(plot_data,bins=10, color=colors)
    plt.title("train on dataset "+ dataset_name[idx])
    plt.legend(labels)
    plt.show()
print("###################################Q3_1#########################################")
