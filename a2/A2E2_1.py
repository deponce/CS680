import numpy as np
import statsmodels.api as sm
import copy
from sklearn import svm

def rescale_to_np_1(data):
    """
    :param data: input, numpy array
    :return: scaled numpy array [-1, 1]
    only for dataset $\y_i \in\{0, 1\}$
    """
    maximum = data.max()
    minimun = data.min()
    return 2*(data-minimun)/(maximum-minimun)-1

def less_1(arr, thr = 1e-10):
    """
    :param arr: input, 1-D numpy array
    :param thr: the thr of the num resolution
    :return: num of entry less than 1 in array
    """
    cnt = 0
    for e in arr:
        if e <= 1+thr:
            cnt = cnt + 1
    return cnt

X_train_A = np.genfromtxt('data/X_train_A.csv', delimiter=',')

Y_train_A = np.genfromtxt('data/Y_train_A.csv', delimiter=',')

soft_margin_svm = svm.SVC(kernel='linear', C=1)

soft_margin_svm.fit(X_train_A, Y_train_A)

coefficient = soft_margin_svm.coef_
y_hat_A = X_train_A.dot(coefficient[0])  # y_hat_A is the prediction on the test set
Y_train_A_scaled = rescale_to_np_1(Y_train_A)  # replace 0's with -1's
p_error = Y_train_A_scaled * y_hat_A
print(p_error)
# p_error: prediction error, in soft-margin SVM
# 1-y\hat{y}<= 0: on the right side of the margin
# 0 <= 1-y\hat{y} <=1 : with in the margin
# y\hat{y < 0 : incorrect

nerror = less_1(p_error,1e-10) 
print("soft-margin SVM weight vector: ", soft_margin_svm.coef_, "\nsoft-margin SVM bias: ", soft_margin_svm.intercept_)
a = copy.deepcopy(soft_margin_svm.coef_)
print("n_sv",soft_margin_svm.support_)




hard_margin_svm = svm.SVC(kernel='linear', C=float('inf'))
hard_margin_svm.fit(X_train_A, Y_train_A)
coefficient = soft_margin_svm.coef_
y_hat_A = X_train_A.dot(coefficient[0])  # y_hat_A is the prediction on the test set
Y_train_A_scaled = rescale_to_np_1(Y_train_A)  # replace 0's with -1's
p_error = Y_train_A_scaled * y_hat_A
print(p_error)

print("hard-margin SVM weight vector: ", hard_margin_svm.coef_, "\nsoft-margin SVM bias: ", hard_margin_svm.intercept_)
print("n_sv",hard_margin_svm.support_)


