import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from matplotlib import pyplot as plt
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
    return np.dot(e.T,e)/n

def get_Lasso_vector(X, Y, reg = 0.0):
    model = Lasso(alpha=reg,tol = 0.0005)
    model.fit(X, Y)
    w = model.coef_
    b = model.intercept_
    return np.r_[w,b]

def get_Ridge_vector(X, Y, reg = 0.0):
    model = Ridge(alpha=reg)
    model.fit(X, Y)
    w = model.coef_
    b = model.intercept_
    return np.r_[w,b]

def get_LinearRegression_vector(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    w = model.coef_
    b = model.intercept_
    return np.r_[w,b]

dataset_name = ['A', 'B', 'C']
num_bins = 200

for idx, X in enumerate(Train_data_X):
    print("dataset: ", dataset_name[idx])
    lrv = get_LinearRegression_vector(X, Train_data_Y[idx])
    print("MES of LinearRegression: ",cal_MSE(lrv[:-1],lrv[-1],Test_data_X[idx],Test_data_Y[idx]))

    r1v = get_Ridge_vector(X, Train_data_Y[idx], 1)
    print("MES of Ridge with reg 1: ",cal_MSE(r1v[:-1],lrv[-1],Test_data_X[idx],Test_data_Y[idx]))

    r10v = get_Ridge_vector(X, Train_data_Y[idx], 10)
    print("MES of Ridge with reg 10: ",cal_MSE(r10v[:-1],lrv[-1],Test_data_X[idx],Test_data_Y[idx]))

    l1v = get_Lasso_vector(X, Train_data_Y[idx], 1/306)
    print("MES of Lasso with reg 1: ",cal_MSE(l1v[:-1],lrv[-1],Test_data_X[idx],Test_data_Y[idx]))

    l10v = get_Lasso_vector(X, Train_data_Y[idx], 10/306)
    print("MES of Lasso with reg 10: ", cal_MSE(l10v[:-1],lrv[-1],Test_data_X[idx],Test_data_Y[idx]))

    plot_data = (lrv[:-1], r1v[:-1], r10v[:-1], l1v[:-1], l10v[:-1])
    colors = ("indigo", "seagreen", "darkslategray", "coral", "firebrick")
    labels = ("LinearRegression", "Ridge with reg 1", "Ridge with reg 10", "Lasso with reg 1", "Lasso with reg 10")
    plt.hist(plot_data,bins=10, color=colors)
    plt.title("train on dataset "+ dataset_name[idx])
    plt.legend(labels)
    plt.show()
print("###################################Q3_1#########################################")
