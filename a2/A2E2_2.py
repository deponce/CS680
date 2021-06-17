import numpy as np
from sklearn import svm
import statsmodels.api as sm
def rescale(data):
    maximum = data.max()
    minimun = data.min()
    return 2*(data-minimun)/(maximum-minimun)-1

def less_1(arr, thr = 1e-10):
    cnt = 0
    for e in arr:
        if e < 1-thr:
            cnt = cnt + 1
    return cnt
X_train_A = np.genfromtxt('data/X_train_A.csv', delimiter=',')

Y_train_A = np.genfromtxt('data/Y_train_A.csv', delimiter=',')
Y_train_A_scaled = rescale(Y_train_A)

clf = svm.SVC(kernel='linear', C=1)

clf.fit(X_train_A, Y_train_A)

# A2E2_2

coefficient = clf.coef_

y_hat_A = X_train_A.dot(coefficient[0])+clf.intercept_

p_error = Y_train_A_scaled*y_hat_A

nerror = less_1(p_error)

print("#<0",nerror)

X_train_B = np.genfromtxt('data/X_train_B.csv', delimiter=',')
Y_train_B = np.genfromtxt('data/Y_train_B.csv', delimiter=',')

X_test_B = np.genfromtxt('data/X_test_B.csv', delimiter=',')
Y_test_B = np.genfromtxt('data/Y_test_B.csv', delimiter=',')

logistic_regression = sm.Logit(Y_train_B, X_train_B).fit()
print("finish fitting Logit")
Y_hat_B = logistic_regression.predict(X_test_B)
Y_error = rescale(Y_hat_B) * rescale(Y_test_B)
n_Y_error = Y_error[Y_error > 0].shape[0]

print(n_Y_error/Y_hat_B.shape[0])

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train_B, Y_train_B)
print("n_sv",clf.n_support_)
print("finish fitting soft margin SVM")

Y_hat_B = clf.predict(X_test_B)
Y_error = rescale(Y_hat_B)* rescale(Y_test_B)
n_Y_error = Y_error[Y_error > 0].shape[0]
print(n_Y_error/Y_hat_B.shape[0])
"""
clf = svm.SVC(kernel='linear', C=float('inf'))
clf.fit(X_train_B, Y_train_B)
print("finish fitting hard margin SVM")
"""


