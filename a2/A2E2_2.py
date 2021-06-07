import numpy as np
from sklearn import svm
import statsmodels.api as sm

X_train_A = np.genfromtxt('data/X_train_A.csv', delimiter=',')

Y_train_A = np.genfromtxt('data/Y_train_A.csv', delimiter=',')

clf = svm.SVC(kernel='linear', C=1)

clf.fit(X_train_A, Y_train_A)

# A2E2_2

coefficient = clf.coef_

y_hat_A = X_train_A.dot(coefficient[0])+clf.intercept_

print(y_hat_A)


X_train_B = np.genfromtxt('data/X_train_B.csv', delimiter=',')
Y_train_B = np.genfromtxt('data/Y_train_B.csv', delimiter=',')

X_test_B = np.genfromtxt('data/X_test_B.csv', delimiter=',')
Y_test_B = np.genfromtxt('data/Y_test_B.csv', delimiter=',')

sm.Logit(Y_train_B, X_train_B).fit()
print("finish fitting Logit")

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train_B, Y_train_B)
print("n_sv",clf.n_support_)
print("finish fitting soft margin SVM")

clf = svm.SVC(kernel='linear', C=float('inf'))
clf.fit(X_train_B, Y_train_B)
print("finish fitting hard margin SVM")

